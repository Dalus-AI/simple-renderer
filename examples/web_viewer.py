from renderer import SimpleRenderer
import argparse
import torch
import numpy as np

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

    # Simple interactive web viewer
    renderer = SimpleRenderer(args)
    renderer.run()
    
    # Or, set custom parameters
    # Comment the above two lines and uncomment the lines below
    # H, W = 640, 852

    # K = np.array([
    #     [403.41072327665165, 0, W/2],
    #     [0, 404.223883786868, H/2],
    #     [0.0, 0.0, 1.0]
    # ])
    
    # renderer = SimpleRenderer(args, override_K=K, H=H, W=W, distortion_params=D)
    # # To run the web viewer
    # renderer.run()