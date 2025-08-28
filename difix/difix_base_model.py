import os
import requests
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch._inductor import config as inductor_config
import torch._dynamo as dynamo
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from peft import LoraConfig
# p = "src/"
# sys.path.append(p)
from einops import rearrange, repeat

from time import time


def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = [t.detach().clone() for t in l_blocks]
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample.add_(skip_in)
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def download_url(url, outf):
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(outf, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print(f"Downloaded successfully to {outf}")
    else:
        print(f"Skipping download, {outf} already exists")


def load_ckpt_from_state_dict(net_difix, optimizer, pretrained_path):
    sd = torch.load(pretrained_path, map_location="cpu")

    if "state_dict_vae" in sd:
        _sd_vae = net_difix.vae.state_dict()
        for k in sd["state_dict_vae"]:
            _sd_vae[k] = sd["state_dict_vae"][k]
        net_difix.vae.load_state_dict(_sd_vae)
    _sd_unet = net_difix.unet.state_dict()
    for k in sd["state_dict_unet"]:
        _sd_unet[k] = sd["state_dict_unet"][k]
    net_difix.unet.load_state_dict(_sd_unet)

    optimizer.load_state_dict(sd["optimizer"])

    return net_difix, optimizer


def save_ckpt(net_difix, optimizer, outf):
    sd = {}
    sd["vae_lora_target_modules"] = net_difix.target_modules_vae
    sd["rank_vae"] = net_difix.lora_rank_vae
    sd["state_dict_unet"] = net_difix.unet.state_dict()
    sd["state_dict_vae"] = {k: v for k, v in net_difix.vae.state_dict().items() if "lora" in k or "skip" in k}

    sd["optimizer"] = optimizer.state_dict()

    torch.save(sd, outf)


class Difix(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_vae=4, mv_unet=False, timestep=999):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()
        self._text_cache = {}

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        vae.config.force_upcast = False

        if mv_unet:
            from mv_unet import UNet2DConditionModel
        else:
            from diffusers import UNet2DConditionModel

        # Prefer TF32 for FP32 paths (harmless if unsupported)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # good for fixed-size inference

        # Choose fast dtype
        self.dtype = torch.float16

        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            target_modules_vae = []

            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]

            target_modules = []
            for id, (name, param) in enumerate(vae.named_modules()):
                if 'decoder' in name and any(name.endswith(x) for x in target_modules_vae):
                    target_modules.append(name)
            target_modules_vae = target_modules
            vae.encoder.requires_grad_(False)

            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae

        # unet.enable_xformers_memory_efficient_attention()
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            unet.set_attn_processor(AttnProcessor2_0())  # uses torch SDPA/Flash attention
        except Exception:
            # Fallback if SDPA not available
            try:
                unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print("Efficient attention not enabled:", e)

        unet.to("cuda", dtype=self.dtype, memory_format=torch.channels_last)
        vae.to("cuda",  dtype=self.dtype, memory_format=torch.channels_last)
        self.text_encoder.to("cuda")  # keep text encoder in fp32 for stability by default

        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([timestep], device="cuda").long()
        self.text_encoder.requires_grad_(False)
        torch._dynamo.config.verbose = True
        torch.backends.cudnn.benchmark = True

        try:
            compile_mode = "max-autotune"   # or "reduce-overhead". Avoid "max-autotune" if first-run is too slow
            self.unet        = torch.compile(self.unet,        mode=compile_mode, fullgraph=False)
            self.vae.encoder = torch.compile(self.vae.encoder, mode="reduce-overhead", fullgraph=False)
            self.vae.decoder = torch.compile(self.vae.decoder, mode="reduce-overhead", fullgraph=False)
            print("torch.compile enabled")
        except Exception as e:
            print("torch.compile skipped:", e)

        # print number of trainable parameters
        print("="*50)
        print(f"Number of trainable parameters in UNet: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Number of trainable parameters in VAE: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6:.2f}M")
        print("="*50)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.unet.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, x, timesteps=None, prompt=None, prompt_tokens=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"
        assert (timesteps is None) != (self.timesteps is None), "Either timesteps or self.timesteps should be provided"

        # Text encoding in FP32 for numerical stability, then cast
        if prompt is not None:
            key = ("prompt", prompt)
            if key not in self._text_cache:
                with torch.inference_mode():
                    ids = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                        padding="max_length", truncation=True, return_tensors="pt").input_ids.to("cuda")
                    self._text_cache[key] = self.text_encoder(ids)[0]
            caption_enc = self._text_cache[key]
        else:
            key = ("tokens", tuple(prompt_tokens.flatten().tolist()))
            if key not in self._text_cache:
                with torch.inference_mode():
                    self._text_cache[key] = self.text_encoder(prompt_tokens.to("cuda"))[0]
            caption_enc = self._text_cache[key]

        if x.dim() == 4:
            # V=1 case, no repeat needed
            v = 1
            caption_enc = caption_enc.to(self.dtype)
        else:
            # V>1 only if you passed a 5-D tensor
            v = x.shape[1]
            caption_enc = repeat(caption_enc, 'b n c -> (b v) n c', v=v).to(self.dtype)
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
            if x.dim() == 5:
                x = rearrange(x, 'b v c h w -> (b v) c h w', v=v)
            # ensure 4-D channels_last reaches compiled modules
            x = x.contiguous(memory_format=torch.channels_last)
            # Using mode() is a tiny speedup over sampling and removes RNG sync
            z = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor
            z = z.contiguous(memory_format=torch.channels_last)
            model_pred = self.unet(z, self.timesteps, encoder_hidden_states=caption_enc).sample
            z_denoised = self.sched.step(model_pred, self.timesteps, z, return_dict=True).prev_sample
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample
        output_image = output_image.float().clamp_(-1, 1)
        if v == 1:
            output_image = output_image.unsqueeze(1)  # make shape [b,1,c,h,w] to match your later [:, 0]
        else:
            output_image = rearrange(output_image, '(b v) c h w -> b v c h w', v=v)
        return output_image

    def sample(self, image, width, height, ref_image=None, timesteps=None, prompt=None, prompt_tokens=None):
        # Handle both PIL images and tensors
        if isinstance(image, torch.Tensor):
            # Expect tensor to be in correct shape [C, H, W] with values in [0, 1] range
            print(f"LF_DEBUG: image.device.type: {image.device.type}. image.shape: {image.shape}")
            input_height, input_width = image.shape[1], image.shape[2]
        else:
            # Handle PIL image as before
            input_width, input_height = image.size

            T = transforms.Compose([
                transforms.Resize((height, width), interpolation=Image.LANCZOS),
                transforms.ToTensor(),
            ])
            image = T(image)

        start = time()
        normalize = transforms.Normalize([0.5], [0.5])
        if ref_image is None:
            # 4-D NCHW, so we can use channels_last
            x = (normalize(image).unsqueeze(0)
                .contiguous(memory_format=torch.channels_last))  # [1, C, H, W]
            if x.device.type == 'cpu':
                x = x.pin_memory().to("cuda", non_blocking=True)
        else:
            # Only build 5-D if you really have 2 views
            x = torch.stack([T(image), T(ref_image)], dim=0)                 # [2, C, H, W]
            x = x.unsqueeze(0).to("cuda", non_blocking=True)                 # [1, 2, C, H, W]
        print(f"##### INPUT SHAPE: {x.shape} #######")
        torch.compiler.cudagraph_mark_step_begin()
        output_image = self.forward(x, timesteps, prompt, prompt_tokens)[:, 0] * 0.5 + 0.5
        end = time()
        print(f"########## Forward pass took {(end-start)*1e3} ms.")
        output_pil = transforms.ToPILImage()(output_image[0].cpu())
        output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)

        return output_pil

    def save_model(self, outf, optimizer):
        sd = {}
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}

        sd["optimizer"] = optimizer.state_dict()

        torch.save(sd, outf)