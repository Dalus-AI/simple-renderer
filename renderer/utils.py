import numpy as np
import torch 
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import multiprocessing as mp

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

def process_gaussians(start_idx, end_idx, vert, SH_C0):
    """Process a subset of the Gaussian splats."""
    chunk_size = end_idx - start_idx
    positions = np.zeros((chunk_size, 3), dtype=np.float32)
    scales = np.zeros((chunk_size, 3), dtype=np.float32)
    rots = np.zeros((chunk_size, 4), dtype=np.float32)
    colors = np.zeros((chunk_size, 4), dtype=np.float32)
    opacities = np.zeros(chunk_size, dtype=np.float32)

    for idx, i in enumerate(range(start_idx, end_idx)):
        v = vert[i]
        positions[idx] = [v["x"], v["y"], v["z"]]
        scales[idx] = np.exp([v["scale_0"], v["scale_1"], v["scale_2"]])
        rots[idx] = [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]
        norm = np.linalg.norm(rots[idx], ord=2, axis=-1, keepdims=True)
        rots[idx] = rots[idx] / norm  # Normalize quaternion

        colors[idx] = [
            0.5 + SH_C0 * v["f_dc_0"],
            0.5 + SH_C0 * v["f_dc_1"],
            0.5 + SH_C0 * v["f_dc_2"],
            1 / (1 + np.exp(-v["opacity"]))
        ]
        opacities[idx] = 1 / (1 + np.exp(-v["opacity"]))  # Sigmoid function

    return positions, scales, rots, colors, opacities

def load_ply_data(file_path):
    device = 'cuda:0'
    print("Reading file...")
    plydata = PlyData.read(file_path)
    print("Done.")
    
    vert = plydata["vertex"]
    
    N = len(vert)
    SH_C0 = 0.28209479177387814
    
    # Parallel Processing Setup
    num_processes = mp.cpu_count()  # Use all available CPU cores
    chunk_size = (N + num_processes - 1) // num_processes  # Split indices evenly
    pool = mp.Pool(processes=num_processes)
    
    print("Parsing Gaussian Splat in parallel...")
    results = [
        pool.apply_async(process_gaussians, (i, min(i + chunk_size, N), vert, SH_C0))
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
        device = torch.device('cuda')
        free_memory = torch.cuda.mem_get_info(device)[0]  # Free memory in bytes
        free_memory_gb = free_memory / (1024 ** 3)  # Convert to GB
        print(f"Free GPU memory: {free_memory_gb:.2f} GB")

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

def color_surface_normal(normal: torch.tensor, mask: torch.tensor=None) -> np.array:
    """
    Visualize surface normal. Transfer surface normal value from [-1, 1] to [0, 255]
    Aargs:
        normal (torch.tensor, [h, w, 3]): surface normal
        mask (torch.tensor, [h, w]): valid masks
    """
    normal = normal.cpu().numpy().squeeze()
    n_img_L2 = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    n_img_norm = normal / (n_img_L2 + 1e-8)
    normal_vis = n_img_norm * 127
    normal_vis += 128
    normal_vis = normal_vis.astype(np.uint8)
    if mask is not None:
        mask = mask.cpu().numpy().squeeze()
        normal_vis[~mask] = 0
    return normal_vis

def save_point_cloud_as_ply(points, filename="output.ply"):
    """Save the point cloud to a PLY file using plyfile."""
    vertex = np.array([(p[0], p[1], p[2]) for p in points], 
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    ply = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)
    ply.write(filename)

def center_pad_image(image, target_width, target_height):
    """
    Creates a blank image of target size and places the given image at the center.
    
    Args:
        image: Input image as a numpy array.
        target_width: Desired width of the output image.
        target_height: Desired height of the output image.
    
    Returns:
        padded_image: The new image with the input image centered.
    """
    img_height, img_width = image.shape[:2]

    if target_width < img_width or target_height < img_height:
        return image
    
    # Create a blank image (black) of the target size
    padded_image = np.zeros((target_height, target_width, 3), dtype=image.dtype)
    
    # Compute the top-left coordinates for centering
    y_offset = (target_height - img_height) // 2
    x_offset = (target_width - img_width) // 2

    # Place the image in the center of the blank image
    padded_image[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = image

    return padded_image

def save_depth_as_pcd(depth_image, K, pcd_filename, project_3d=True):
    """
    Save a depth image as a .pcd file.
    
    Parameters:
        depth_image (np.ndarray): The depth image (H x W) where each pixel contains a depth value in meters.
        K (np.ndarray): Intrinsic camera matrix (3x3).
        pcd_filename (str): Output filename (should end in .pcd).
        project_3d (bool): If True, convert to full 3D (X, Y, Z) coordinates. If False, save as (u, v, Z).
    """
    if len(depth_image.shape) == 3:
        depth_image = depth_image[:,:,0]
    H, W = depth_image.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    points = []
    
    for v in range(H):
        for u in range(W):
            z = depth_image[v, u]
            if z == 0:  # Skip invalid depth values
                continue

            if project_3d:
                # Back-project to 3D space using intrinsics
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
            else:
                # Save only (u, v, z)
                points.append([u, v, z])

    points = np.array(points)
    
    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Save PCD file
    o3d.io.write_point_cloud(pcd_filename, pcd)
    print(f"Saved {len(points)} points to {pcd_filename}")
