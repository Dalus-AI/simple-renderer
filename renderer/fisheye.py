import enum

import cv2
import numpy as np
import scipy.interpolate
import open3d as o3d

class DistortMode(enum.Enum):
    LINEAR = 'linear'
    NEAREST = 'nearest'


def distort_image(img: np.ndarray, cam_intr: np.ndarray, dist_coeff: np.ndarray,
                  mode: DistortMode = DistortMode.LINEAR, crop_output: bool = True,
                  crop_type: str = "corner") -> np.ndarray:
    """Apply fisheye distortion to an image

    Args:
        img (numpy.ndarray): BGR image. Shape: (H, W, 3)
        cam_intr (numpy.ndarray): The camera intrinsics matrix, in pixels: [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]
                            Shape: (3, 3)
        dist_coeff (numpy.ndarray): The fisheye distortion coefficients, for OpenCV fisheye module.
                            Shape: (1, 4)
        mode (DistortMode): For distortion, whether to use nearest neighbour or linear interpolation.
                            RGB images = linear, Mask/Surface Normals/Depth = nearest
        crop_output (bool): Whether to crop the output distorted image into a rectangle. The 4 corners of the input
                            image will be mapped to 4 corners of the distorted image for cropping.
        crop_type (str): How to crop.
            "corner": We crop to the corner points of the original image, maintaining FOV at the top edge of image.
            "middle": We take the widest points along the middle of the image (height and width). There will be black
                      pixels on the corners. To counter this, original image has to be higher FOV than the desired output.

    Returns:
        numpy.ndarray: The distorted image, same resolution as input image. Unmapped pixels will be black in color.
    """
    assert cam_intr.shape == (3, 3)
    assert dist_coeff.shape == (4,)

    imshape = img.shape
    if len(imshape) == 3:
        h, w, chan = imshape
    elif len(imshape) == 2:
        h, w = imshape
        chan = 1
    else:
        raise RuntimeError(f'Image has unsupported shape: {imshape}. Valid shapes: (H, W), (H, W, N)')

    imdtype = img.dtype

    # Get array of pixel co-ords
    xs = np.arange(w)
    ys = np.arange(h)
    xv, yv = np.meshgrid(xs, ys)
    img_pts = np.stack((xv, yv), axis=2)  # shape (H, W, 2)
    img_pts = img_pts.reshape((-1, 1, 2)).astype(np.float32)  # shape: (N, 1, 2)

    # Get the mapping from distorted pixels to undistorted pixels
    undistorted_px = cv2.fisheye.undistortPoints(img_pts, cam_intr, dist_coeff)  # shape: (N, 1, 2)
    undistorted_px = cv2.convertPointsToHomogeneous(undistorted_px)  # Shape: (N, 1, 3)
    undistorted_px = np.tensordot(undistorted_px, cam_intr, axes=(2, 1))  # To camera coordinates, Shape: (N, 1, 3)
    undistorted_px = cv2.convertPointsFromHomogeneous(undistorted_px)  # Shape: (N, 1, 2)
    undistorted_px = undistorted_px.reshape((h, w, 2))  # Shape: (H, W, 2)
    undistorted_px = np.flip(undistorted_px, axis=2)  # flip x, y coordinates of the points as cv2 is height first

    # Map RGB values from input img using distorted pixel co-ordinates
    if chan == 1:
        img = np.expand_dims(img, 2)
    interpolators = [scipy.interpolate.RegularGridInterpolator((ys, xs), img[:, :, channel], method=mode.value,
                                                               bounds_error=False, fill_value=0)
                     for channel in range(chan)]
    img_dist = np.dstack([interpolator(undistorted_px) for interpolator in interpolators])

    if imdtype == np.uint8:
        # RGB Image
        img_dist = img_dist.round().clip(0, 255).astype(np.uint8)
    elif imdtype == np.uint16:
        # Mask
        img_dist = img_dist.round().clip(0, 65535).astype(np.uint16)
    elif imdtype == np.float16 or imdtype == np.float32 or imdtype == np.float64:
        img_dist = img_dist.astype(imdtype)
    else:
        raise RuntimeError(f'Unsupported dtype for image: {imdtype}')

    if crop_output:
        # Crop rectangle from resulting distorted image
        # Get mapping from undistorted to distorted
        distorted_px = cv2.convertPointsToHomogeneous(img_pts)  # Shape: (N, 1, 3)
        cam_intr_inv = np.linalg.inv(cam_intr)
        distorted_px = np.tensordot(distorted_px, cam_intr_inv, axes=(2, 1))  # To camera coordinates, Shape: (N, 1, 3)
        distorted_px = cv2.convertPointsFromHomogeneous(distorted_px)  # Shape: (N, 1, 2)
        distorted_px = cv2.fisheye.distortPoints(distorted_px, cam_intr, dist_coeff)  # shape: (N, 1, 2)
        distorted_px = distorted_px.reshape((h, w, 2))
        if crop_type == "corner":
            # Get the corners of original image. Round values up/down accordingly to avoid invalid pixel selection.
            top_left = np.ceil(distorted_px[0, 0, :]).astype(np.int32)
            bottom_right = np.floor(distorted_px[(h - 1), (w - 1), :]).astype(np.int32)
            img_dist = img_dist[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
        elif crop_type == "middle":
            # Get the widest point of original image, then get the corners from that.
            width_min = np.ceil(distorted_px[int(h / 2), 0, 0]).astype(np.int32)
            width_max = np.ceil(distorted_px[int(h / 2), -1, 0]).astype(np.int32)
            height_min = np.ceil(distorted_px[0, int(w / 2), 1]).astype(np.int32)
            height_max = np.ceil(distorted_px[-1, int(w / 2), 1]).astype(np.int32)
            img_dist = img_dist[height_min:height_max, width_min:width_max]
        else:
            raise ValueError

    if chan == 1:
        img_dist = img_dist[:, :, 0]

    return img_dist

def save_depth_as_pcd_fisheye(
    depth_image: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    pcd_filename: str,
    project_3d: bool = True
):
    """
    Save a fisheye camera depth image as a .pcd file, either as raw (u, v, Z) or
    projected 3D points (X, Y, Z).

    Args:
        depth_image (np.ndarray): 2D array (H x W) of depth values (in meters).
        K (np.ndarray): 3x3 intrinsic camera matrix for fisheye camera.
        D (np.ndarray): Distortion coefficients for fisheye camera (1x4 or 4,).
        pcd_filename (str): Output .pcd file path.
        project_3d (bool): 
            - If True, store 3D points in camera coordinates: (X, Y, Z).
            - If False, store raw pixel coords + depth: (u, v, Z).
    """
    # Ensure correct shapes
    # breakpoint()
    if len(depth_image.shape) == 3:
        depth_image = depth_image[:,:,0]
    elif len(depth_image.shape) == 4:
        depth_image = depth_image[0, :, :, 0]
    assert depth_image.ndim == 2, "Depth image must be HxW."
    assert K.shape == (3, 3), "Intrinsic matrix K must be 3x3."
    assert D.shape[0] == 4, "Fisheye distortion must have 4 coefficients."

    H, W = depth_image.shape

    # Create a list of all pixel coordinates: shape (N, 1, 2)
    # (xv, yv) = pixel coordinates in standard indexing (column=x, row=y)
    xv, yv = np.meshgrid(np.arange(W), np.arange(H))
    pixel_coords = np.stack((xv, yv), axis=-1).astype(np.float32).reshape(-1, 1, 2)

    # Flatten depth to match these pixel coordinates
    depth_flat = depth_image.reshape(-1)

    # We only want valid (non-zero) depth
    valid_mask = (depth_flat > 0)
    pixel_coords_valid = pixel_coords[valid_mask]
    depth_valid = depth_flat[valid_mask]

    if project_3d:
        # For fisheye cameras, we "undistort" the pixel to its ideal normalized ray
        # shape of undistorted_pts -> (N, 1, 2)
        undistorted_pts = cv2.fisheye.undistortPoints(pixel_coords_valid, K, D)

        # undistortPoints() gives us normalized image coords (x, y) in the pinhole sense
        # The corresponding 3D points are then: (x*z, y*z, z)
        # We'll do this in a vectorized way:

        # Nx2 -> Nx3
        x_norm = undistorted_pts[:, 0, 0]
        y_norm = undistorted_pts[:, 0, 1]
        z_vals = depth_valid

        X = x_norm * z_vals
        Y = y_norm * z_vals
        Z = z_vals

        points_3d = np.stack([X, Y, Z], axis=-1)
        points = points_3d
    else:
        # Just store (u, v, z) in pixel space
        u_vals = pixel_coords_valid[:, 0, 0]
        v_vals = pixel_coords_valid[:, 0, 1]
        z_vals = depth_valid
        points = np.stack([u_vals, v_vals, z_vals], axis=-1)

    # Convert to float64 for Open3D
    points = points.astype(np.float64)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Save as PCD
    o3d.io.write_point_cloud(pcd_filename, pcd)
    print(f"Saved {len(points)} points to {pcd_filename}")