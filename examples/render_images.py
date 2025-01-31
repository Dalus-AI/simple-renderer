from renderer import SimpleRenderer
import argparse
import torch
import numpy as np
from imageio import imsave
from renderer.utils import save_depth_as_pcd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument("--backend", type=str, default="2dgs", help="3dgs, 2dgs")
    parser.add_argument(
        "--ply", type=str, nargs="+", default=["assets/lucky_cat.ply"], help="path to the .ply file(s)"
    )

    args = parser.parse_args()

    assert torch.cuda.is_available(), "Renderer requires a GPU to run. PyTorch cannot detect a GPU."

    H, W = 640, 852

    K = np.array([
        [403.41072327665165, 0, W/2],
        [0, 404.223883786868, H/2],
        [0.0, 0.0, 1.0]
    ])
    
    # Fix a position and rotation for the camera.
    # This T is the world-to-camera transform
    T = np.array([[-8.7352e-01,  4.8679e-01,  5.7073e-09, -4.0487e-01],
        [ 1.8977e-01,  3.4053e-01, -9.2088e-01,  3.9384e-02],
        [-4.4828e-01, -8.0441e-01, -3.8984e-01,  1.2369e+00],
        [ 2.3212e-08, -3.4397e-08, -2.3190e-09,  1.0000e+00]])
    
    renderer = SimpleRenderer(args)
    # To just render an image
    rgb_im, depth_im, metadata = renderer.simple_render_fn(
        T, K, (W, H), get_raw_depth=True
    )
    imsave("rgb_image.png", rgb_im)
    imsave("depth_image.png", depth_im)
    save_depth_as_pcd(metadata["depth_raw"].cpu().numpy(), K, "depth.pcd")