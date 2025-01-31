import argparse
import torch 
import numpy as np
from imageio import imsave 

from renderer.simple_renderer import SimpleRenderer
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
        [403.41072327665165, 0, W//2],
        [0, 404.223883786868, H//2],
        [0.0, 0.0, 1.0]
    ])

    # Distortion parameters for a fisheye camera, for example
    # Leave as None for cameras without distortion
    D = np.array(
        [-0.032504882829736806,
        -0.0037828549055173716,
        -1.7465655639965706e-05,
        -0.00030060999584025777]).reshape((4,1))
    
    # Fix a position and rotation for the camera. This will disable the interactivity of the web viewer
    # This T is the world-to-camera transform
    # Leave T as None to only fix the intrinsics and keep the camera's position and rotation interactive

    T = np.array([[ 5.1012e-01,  8.6010e-01,  1.7492e-09, -6.7338e-01],
        [ 1.3904e-01, -8.2464e-02, -9.8685e-01,  9.5698e-02],
        [-8.4879e-01,  5.0341e-01, -1.6166e-01,  6.6488e-01],
        [-6.1578e-08, -1.1769e-07,  6.5397e-10,  1.0000e+00]])
    
    renderer = SimpleRenderer(args, override_K=K, override_T=None, H=H, W=W, distortion_params=D)
    # To run the web viewer
    # renderer.run(stream_T=False)
    # To just render an image
    rgb_im, depth_im, metadata = renderer.simple_render_fn(
        T, K, (W, H), get_raw_depth=True
    )
    imsave("rgb_image.png", rgb_im)
    imsave("depth_image.png", depth_im)
    # Save depth pcd
    save_depth_as_pcd(metadata["depth_raw"].cpu().numpy(), K, "depth.pcd")