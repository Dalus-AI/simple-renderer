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
from tqdm import trange
import viser
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R

from gsplat.distributed import cli
from gsplat.rendering import rasterization, rasterization_2dgs
from plyfile import PlyData

def rotate_splat_from_euler(positions, quats, angle_rotations, degrees=True, as_tensor=False):
    print("Rotating splat")
    rotation_matrix = None
    for (axis, mag) in angle_rotations:
            if rotation_matrix is not None:
                    rotation_matrix = R.from_euler(axis, mag, degrees=degrees) * rotation_matrix
            else:
                    rotation_matrix = R.from_euler(axis, mag, degrees=degrees)
    # 1: Rotate points
    new_positions = positions @ rotation_matrix.as_matrix().T
    # Step 2: Create a rotation matrix for a 90-degree rotation around the x-axis
    # Rotate the random quaternions using the rotation matrix
    new_quats = (rotation_matrix * R.from_quat(quats, scalar_first=True)).as_quat(scalar_first=True)
    if as_tensor:
          new_positions = _convert_to_tensor(new_positions)
          new_quats = _convert_to_tensor(new_quats)
    print("done.")
    return new_positions, new_quats

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

def process_gaussians(thread_idx, start_idx, end_idx, vert, SH_C0):
    """Process a subset of the Gaussian splats."""
    chunk_size = end_idx - start_idx
    positions = np.zeros((chunk_size, 3), dtype=np.float32)
    scales = np.zeros((chunk_size, 3), dtype=np.float32)
    rots = np.zeros((chunk_size, 4), dtype=np.float32)
    opacities = np.zeros(chunk_size, dtype=np.float32)
    use_sh = True
    try:
        _ = vert[0]["f_rest_1"]
    except ValueError:
        use_sh = False
    
    if use_sh:
        colors = np.zeros((chunk_size, 16, 3), dtype=np.float32)
    else:
        colors = np.zeros((chunk_size, 4), dtype=np.float32)

    batch_gs_idx = 0
    
    for global_gs_idx in trange(start_idx, end_idx, position=thread_idx):
        v = vert[global_gs_idx]
        positions[batch_gs_idx] = [v["x"], v["y"], v["z"]]
        scales[batch_gs_idx] = np.exp([v["scale_0"], v["scale_1"], v["scale_2"]])
        rots[batch_gs_idx] = [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]
        norm = np.linalg.norm(rots[batch_gs_idx], ord=2, axis=-1, keepdims=True)
        rots[batch_gs_idx] = rots[batch_gs_idx] / norm  # Normalize quaternion
        
        if use_sh: # Use SH
            colors_raw = np.array([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]] + [v[f"f_rest_{i}"] for i in range(45)], dtype=np.float32)
            colors[batch_gs_idx] = np.reshape(colors_raw, (16, 3))
            # scale_0 = lambda x: 0.5 + SH_C0 * x
            # colors[batch_gs_idx][0] = [scale_0(v["f_dc_0"]), scale_0(v["f_dc_1"]), scale_0(v["f_dc_2"])]
            
            # SH_CIs = np.array([SH_C1]*3 + SH_C2 + SH_C3, dtype=np.float32)
            # for sh_idx in range(0, 45, 3):
            #     colors[batch_gs_idx][sh_idx//3+1] = [v[f"f_rest_{sh_idx}"]*SH_CIs[sh_idx//3], v[f"f_rest_{sh_idx+1}"]*SH_CIs[sh_idx//3], v[f"f_rest_{sh_idx+2}"]*SH_CIs[sh_idx//3]]
        else:
            colors[batch_gs_idx] = [
                0.5 + SH_C0 * v["f_dc_0"],
                0.5 + SH_C0 * v["f_dc_1"],
                0.5 + SH_C0 * v["f_dc_2"],
                1 / (1 + np.exp(-v["opacity"]))
            ]
        opacities[batch_gs_idx] = 1 / (1 + np.exp(-v["opacity"]))  # Sigmoid function
        batch_gs_idx += 1

    return positions, scales, rots, colors, opacities

def load_ply_data(file_path):
    device = 'cuda:0'
    plydata = PlyData.read(file_path)
    vert = plydata["vertex"]
        
    N = len(vert)
    SH_C0 = 0.28209479177387814
    
    # Parallel Processing Setup
    num_processes = mp.cpu_count()  # Use all available CPU cores
    chunk_size = (N + num_processes - 1) // num_processes  # Split indices evenly
    pool = mp.Pool(processes=num_processes)
    
    print("Parsing Gaussian Splat in parallel...")
    results = [
        pool.apply_async(process_gaussians, (i // chunk_size, i, min(i + chunk_size, N), vert, SH_C0))
        for i in range(0, N, chunk_size)
    ]
    
    pool.close()
    pool.join()
    
    # Gather results
    positions, scales, rots, colors, opacities = zip(*[r.get() for r in results])
    
    # Concatenate all chunks
    positions = np.vstack(positions)
    scales = np.vstack(scales)
    rots = np.vstack(rots)
    colors = np.vstack(colors)
    opacities = np.hstack(opacities)
    
    # Convert to tensors
    positions = _convert_to_tensor(positions, device)
    rots = _convert_to_tensor(rots, device)
    scales = _convert_to_tensor(scales, device)
    colors = _convert_to_tensor(colors, device)
    opacities = _convert_to_tensor(opacities, device)    
    return positions, rots, scales, colors, opacities

def print_free_gpu_space():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        free_memory = torch.cuda.mem_get_info(device)[0]  # Free memory in bytes
        free_memory_gb = free_memory / (1024 ** 3)  # Convert to GB
        print(f"Free GPU memory: {free_memory_gb:.2f} GB")
        
def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)
    server = viser.ViserServer(port=args.port, verbose=False)
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
    sh_degree = None if len(colors.shape) == 2 else int(np.sqrt(colors.shape[1] - 1))

    if int(args.rotate) == 1:
        means, quats = rotate_splat_from_euler(means.cpu().numpy(), quats.cpu().numpy(), angle_rotations=[('x', 180.0)], as_tensor=True)
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

    
    _ = nerfview.Viewer(
        server=server,
        render_fn=nerfview_render_fn,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument("--backend", type=str, default="3dgs", help="3dgs, 2dgs")
    parser.add_argument(
        "--ply", type=str, nargs="+", default=None, help="path to the .ply file(s)", required=True
    )
    parser.add_argument(
        "--rotate", type=int, default=1, help="rotate around x by 180", required=False)

    args = parser.parse_args()

    cli(main, args, verbose=True)
