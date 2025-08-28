"""
Inference only model file for Difix Model
"""

import os
import torch
from PIL import Image
from .difix_base_model import Difix
from .utils import gpu_mem_report


os.environ["TOKENIZERS_PARALLELISM"] = "false"


PROMPT = "remove degradation"
MODEL_NAME = None
MODEL_PATH = "/data/diffmodel/model_16001.pkl"
TIMESTEP = 199
MV_UNET = False  # No reference image (mv_unet off)
WARMUP_IMAGES = [
    "/root/parallax/easy.jpg",
    "/root/parallax/mid.jpg",
    "/root/parallax/hard.jpg",
]
WARMUP_REPEATS = 3  # Number of times to run through the WARMUP_IMAGES


import time

class GSplatDifixModel(torch.nn.Module):

    def _should_slow_log(self) -> bool:
        """
        Flag to log after N-seconds to avoid over-logging of processes that happen
        multiple times per second.
        """
        if time.time() - self._last_log_epoch < self._slow_log_interval:
            return False
        self._last_log_epoch = time.time()
        return True

    def __init__(self):
        super().__init__()

        # slow logging interval
        self._slow_log_interval = 30.  # N-seconds interval
        self._last_log_epoch = time.time() - (self._slow_log_interval + 1)

        # Enable performance optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Load the Difix model with given constants
        print("Loading Difix model...")
        gpu_mem_report("Before model load.")

        self.model = Difix(
            pretrained_name=MODEL_NAME,
            pretrained_path=MODEL_PATH,
            timestep=TIMESTEP,
            mv_unet=MV_UNET,
        )
        self.model.set_eval()  # Set and keep model in eval mode

        gpu_mem_report("After model load.")
        print("Model loaded.")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        This function takes multiple images in Tensor format.
        `input_tensor` must be B, H, W, C where B is the batch size.

        WARN! This inference function is very sensitive to the way we use rasterizer(..) (mainly our choice to use RGB only):
        The shape `H, W, C` comes from gsplat's rasterizer(..) output which is in a channel-last format.
            https://github.com/nerfstudio-project/gsplat/blob/65042cc501d1cdbefaf1d6f61a9a47575eec8c71/gsplat/rendering.py#L230
                Our C = gsplats' D, which refers to channels, RGB in our case.

        In this function, each entry in the batch is a distinct image from the scene or different scenes, it not multiple views
        of the same inference view-area.
        """

        assert input_tensor.is_cuda, "Input tensor must be on CUDA"
        assert input_tensor.ndim == 4, f"Input tensor must have 4 dimensions (B, H, W, C), got {input_tensor.ndim}"

        if self._should_slow_log():
            print(f"LF_DEBUG: input_tensor.shape: {input_tensor.shape}")

        # Get batch size
        batch_size = input_tensor.shape[0]  # Number of distinct scene images

        # Pre-allocate output list for better memory efficiency
        outputs = [None] * batch_size
        # NOTE: We run inference in sequence per each scene-area image.
        for i in range(batch_size):
            # Extract one image (H, W, C) in channel-last format
            img = input_tensor[i]  # shape: (H, W, C)
            H_in, W_in, C = img.shape  # original dimensions and channels

            # Permute to channel-first for model, shape: (C, H, W)
            img_c_first = img.permute(2, 0, 1)

            # Normalize to [-1, 1]
            img_norm = img_c_first * 2.0 - 1.0

            # Run the diffusion model (Difix) inference
            # Prepare a batch of size 1 for the model -- this puts us in the "# V=1 case, no repeat needed" path of Difix's forward
            img_batch = img_norm.unsqueeze(0)  # shape: (1, C, H_model, W_model)
            # Ensure optimal memory format for GPU performance
            img_batch = img_batch.contiguous(memory_format=torch.channels_last)
            with torch.inference_mode():
                # start_time = time.time()
                # torch.compiler.cudagraph_mark_step_begin()
                # Use no_grad for additional memory savings
                with torch.no_grad():
                    out_tensor = self.model.forward(img_batch, prompt=PROMPT, timesteps=None)
                # if self._should_slow_log():
                #     print(f"LF_DEBUG: img_batch.shape: {img_batch.shape}")
                #     end_time = time.time()
                #     print(f"LF_DEBUG: internal Difix.forward time: {(end_time - start_time)*1e3} ms")
            # The output shape from Difix.forward is (1, 1, C, H_in, W_in)
            out_image = out_tensor[:, 0] * 0.5 + 0.5  # convert to [0,1], shape: (1, C, H_in, W_in)
            out_image = out_image.squeeze(0)          # shape: (C, H_in, W_in)

            # Convert to channel-last format used in sim-stack
            output_img = out_image.permute(1, 2, 0)  # shape: from (C, H, W) -> (H_in, W_in, C)
            outputs[i] = output_img  # Direct assignment instead of append

        # Stack outputs
        result = torch.stack(outputs, dim=0)
        return result