"""Based on simple_viewer.py from the gsplat library

```bash
python examples/simple_viewer.py --scene_grid 13
```
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
import tqdm
import viser

from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization, rasterization_2dgs
from plyfile import PlyData

def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    if args.load_from_ckpt:
        means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
        for ckpt_path in args.ckpt:
            ckpt = torch.load(ckpt_path, map_location=device)["splats"]
            means.append(ckpt["means"])
            quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
            scales.append(torch.exp(ckpt["scales"]))
            opacities.append(torch.sigmoid(ckpt["opacities"]))
            sh0.append(ckpt["sh0"])
            shN.append(ckpt["shN"])
    
    elif args.load_from_ply:
        means, quats, scales, opacities = [], [], [], []
        sh0, shN = [], []  # For spherical harmonics

        for ply_path in args.ply:
            ply_data = PlyData.read(ply_path)

            vertex = ply_data['vertex']
            
            # Extract means (x, y, z)
            means.append(torch.tensor(np.column_stack([
                vertex['x'], vertex['y'], vertex['z']
            ]), device=device, dtype=torch.float32))
            
            # Extract quaternions (w, x, y, z)
            quats.append(torch.tensor(np.column_stack([
                vertex['rot_0'], vertex['rot_1'], 
                vertex['rot_2'], vertex['rot_3']
            ]), device=device, dtype=torch.float32))
            
            # Extract scales
            scales.append(torch.tensor(np.column_stack([
                vertex['scale_0'], vertex['scale_1'], vertex['scale_2']
            ]), device=device, dtype=torch.float32))
            
            # Extract opacities
            opacities.append(torch.tensor(vertex['opacity'], 
                device=device, dtype=torch.float32))
            
            # Extract spherical harmonics
            # First 3 coefficients (DC term)
            sh0.append(torch.tensor(np.column_stack([
                vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']
            ]), device=device, dtype=torch.float32))
            
            # Remaining coefficients (f_rest_0 through f_rest_44)
            if 'f_rest_0' in vertex:
                rest_coeffs = np.column_stack([
                    [vertex[f'f_rest_{i}'] for i in range(45)]
                ]).reshape(-1, 15, 3)  # Reshape to match expected format
                shN.append(torch.tensor(rest_coeffs, device=device, dtype=torch.float32))

    # Concatenate if multiple PLY files
    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    sh0 = torch.cat(sh0, dim=0)
    if len(shN) > 0:
        shN = torch.cat(shN, dim=0)
        colors = torch.cat([sh0, shN], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    else:
        colors = sh0
        sh_degree = None
    print("Number of Gaussians:", len(means))
    print("Using sh degree:", sh_degree)

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
            rasterization_fn = rasterization
        elif args.backend == "2dgs":
            from gsplat import rasterization_inria_wrapper

            rasterization_fn = rasterization_inria_wrapper
        else:
            raise ValueError

        render_colors, render_alphas, meta = rasterization_fn(
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
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        if args.backend == "3dgs":
            rasterization_fn = rasterization
        elif args.backend == "2dgs":
            rasterization_fn = rasterization_2dgs
        else:
            raise ValueError

        render_colors, render_alphas, meta = rasterization_fn(
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
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument("--backend", type=str, default="3dgs", help="3dgs, 2dgs")
    parser.add_argument(
        "--load_from_ckpt", action="store_true", help="load from checkpoint files"
    )
    parser.add_argument(
        "--load_from_ply", action="store_true", help="load from PLY files"
    )
    parser.add_argument(
        "--web_viewer", action="store_true", help="launch web viewer interface"
    )
    parser.add_argument(
        "--ply", type=str, nargs="+", default=None, help="path to the .ply file(s)"
    )

    args = parser.parse_args()

    cli(main, args, verbose=True)