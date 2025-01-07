"""Based on simple_viewer.py from the gsplat library

bash
python examples/simple_viewer.py --scene_grid 13

"""

import argparse
import math
import os
import time
from typing import Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import viser

from gsplat.distributed import cli
from gsplat.rendering import rasterization, rasterization_2dgs
from plyfile import PlyData

def _convert_to_tensor(arr, device='cuda:0'):
    """Convert array to tensor."""
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr).float().to(device)
    if isinstance(arr, torch.Tensor):
        arr.to(device=device)
    if isinstance(arr, list) and isinstance(arr[0], np.ndarray):
        arr = torch.from_numpy(np.array(arr)).float().to(device)
    return arr

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_ply_data(file_path):
    device = 'cuda:0'
    print("reading file...")
    plydata = PlyData.read(file_path)
    print("done.")
    vert = plydata["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )
    N = len(vert)

    positions = np.zeros((N, 3), dtype=np.float32)
    scales = np.zeros((N, 3), dtype=np.float32)
    rots = np.zeros((N, 4), dtype=np.float32)
    colors = np.zeros((N, 4), dtype=np.float32)
    opacities = np.zeros(N, dtype=np.float32)

    SH_C0 = 0.28209479177387814
    print("Parsing Gaussian Splat...")
    for i in tqdm(sorted_indices):
        v = vert[i]
        positions[i] = [v["x"], v["y"], v["z"]]
        scales[i] = np.exp([v["scale_0"], v["scale_1"], v["scale_2"]])
        rots[i] = [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]
        norm = np.linalg.norm(rots[i], ord=2, axis=-1, keepdims=True)
        # Normalize the quaternion
        rots[i] = rots[i] / norm

        colors[i] = [
            0.5 + SH_C0 * v["f_dc_0"],
            0.5 + SH_C0 * v["f_dc_1"],
            0.5 + SH_C0 * v["f_dc_2"],
            1 / (1 + np.exp(-v["opacity"]))
        ]
        opacities[i] = sigmoid(v["opacity"])
    positions = _convert_to_tensor(positions, device)
    rots = _convert_to_tensor(rots, device)
    scales = _convert_to_tensor(scales, device)
    colors = _convert_to_tensor(colors, device)
    opacities = _convert_to_tensor(opacities, device)
    return positions, rots, scales, colors, opacities

def print_free_gpu_space():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        free_memory = torch.cuda.mem_get_info(device)[0]  # Free memory in bytes
        free_memory_gb = free_memory / (1024 ** 3)  # Convert to GB
        print(f"Free GPU memory: {free_memory_gb:.2f} GB")
        
def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)
    
    means, quats, scales, colors, opacities = [], [], [], [], []
    for ply_path in args.ply:
        gs_means, gs_quats, gs_scales, gs_colors, gs_opacities = load_ply_data(ply_path)
        means.append(gs_means)
        quats.append(gs_quats)
        scales.append(gs_scales)
        colors.append(gs_colors)
        opacities.append(gs_opacities)

    # Concatenate if multiple PLY files
    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    colors = torch.cat(colors, dim=0)
    sh_degree = None
    
    print("Number of Gaussians:", len(means))
    print("Using sh degree:", sh_degree)
    print_free_gpu_space()
    
    # register and open viewer
    @torch.no_grad()
    def nerfview_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        """Render a frame of the 3D Gaussian scene from a given camera viewpoint.

        Args:
            camera_state (nerfview.CameraState): Current camera parameters including:
                - c2w: Camera-to-world transformation matrix
                - intrinsics: Camera intrinsic parameters
            img_wh (Tuple[int, int]): Target image dimensions (width, height)

        Returns:
            np.ndarray: Rendered RGB image as a numpy array with shape (height, width, 3)

        The function:
        1. Extracts camera parameters and transforms them to torch tensors
        2. Selects appropriate rasterization backend (gsplat or inria)
        3. Performs Gaussian splatting using the specified backend
        4. Returns the final rendered image
        """
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        if args.backend == "3dgs":
            render_colors, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, S, 3]
                viewmat[None],  # [1, 4, 4]
                K[None],  # [1, 3, 3]
                width,
                height,
                sh_degree=sh_degree,
                render_mode="RGB",
                # this is to speedup large-scale rendering by skipping far-away Gaussians.
                radius_clip=3,
            )
        elif args.backend == "2dgs":
            render_colors, render_alphas, _, _, _, _, _ = rasterization_2dgs(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, 3]
                viewmat[None],  # [1, 4, 4]
                K[None],  # [1, 3, 3]
                width,
                height,
                sh_degree=sh_degree,
                render_mode="RGB",
                # this is to speedup large-scale rendering by skipping far-away Gaussians.
                radius_clip=3,
            )
        else:
            raise ValueError
        render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
        return render_rgbs

    # register and open viewer
    @torch.no_grad()
    def simple_render_fn(T, K, img_wh: Tuple[int, int]):
        """Render a frame of the 3D Gaussian scene from a given camera viewpoint.

        Args:
            T (np array): World-to-camera transformation matrix
            K (np array): Camera intrinsic parameters
            img_wh (Tuple[int, int]): Target image dimensions (width, height)

        Returns:
            np.ndarray: Rendered RGB image as a numpy array with shape (height, width, 3)
        """
        width, height = img_wh
        K = torch.from_numpy(K).float().to(device)
        viewmat = torch.from_numpy(T).float().to(device)

        if args.backend == "3dgs":
            render_colors, _, _ = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, S, 3]
                viewmat[None],  # [1, 4, 4]
                K[None],  # [1, 3, 3]
                width,
                height,
                sh_degree=sh_degree,
                render_mode="RGB",
                # this is to speedup large-scale rendering by skipping far-away Gaussians.
                radius_clip=0,
            )
        elif args.backend == "2dgs":
            render_colors, render_alphas, _, _, _, _, _ = rasterization_2dgs(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, 3]
                viewmat[None],  # [1, 4, 4]
                K[None],  # [1, 3, 3]
                width,
                height,
                sh_degree=sh_degree,
                render_mode="RGB",
                # this is to speedup large-scale rendering by skipping far-away Gaussians.
                radius_clip=0,
            )
        else:
            raise ValueError
        
        render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
        return render_rgbs

    if args.web_viewer:
        server = viser.ViserServer(port=args.port, verbose=False)
        _ = nerfview.Viewer(
            server=server,
            render_fn=nerfview_render_fn,
            mode="rendering",
        )
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(100000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument("--backend", type=str, default="3dgs", help="3dgs, 2dgs")
    parser.add_argument(
        "--web_viewer", action="store_true", help="launch web viewer interface"
    )
    parser.add_argument(
        "--ply", type=str, nargs="+", default=None, help="path to the .ply file(s)", required=True
    )

    args = parser.parse_args()

    cli(main, args, verbose=True)