import time
from typing import Tuple
import warnings

import nerfview
import numpy as np
import torch
import viser
from imageio import imsave

from gsplat.rendering import rasterization, rasterization_2dgs
from renderer.utils import *
from renderer.fisheye import *

class SimpleRenderer:

    def __init__(self, args, override_K=None, override_T=None, H=None, W=None, distortion_params=None):
        """
        Initialize the Renderer

        argrs:
            args: CLI args dict (see __main__ function)
            override_K: Camera intrinsics matrix (3x3 np array), or None. If not None, the renderer will be forced to use this intrinsic matrix.
                        Note: It is highly recommended to set W = int(K[0,-1] * 2.0) and H = int(K[1,-1] * 2.0) if you're overriding K
                        Defaults to None.
            override_T: World-to-camera transformation matrix (4x4 np array) or None. If not None, the renderer will be forced to use this extrinsics matrix.
                        Note: Since this fixes the position and rotation of the camera, this means that the web viewer will no longer be interactive
                        Defaults to None.
            H: int representing the Height of the renderer image, or None. If None, the height of the browser window is used as the image height.
                Note: The using the web_viewer, the rendered image will be resized to fit your browser window
            W: int representing the Width of the renderer image, or None. If None, the width of the browser window is used as the image width.
                Note: The using the web_viewer, the rendered image will be resized to fit your browser window
            distortion_params: fisheye distortion coefficients as a np array, or None. If None, no distortion is applied
        """
        self.device = torch.device("cuda:0")
        self.mode = "rgb"
        self.args = args
        self.K = override_K
        self.T = override_T
        self.distortion_params = distortion_params
        self.H = H
        self.W = W

        self._load_gsplat(args)

    def _add_gui(self, server: viser.ViserServer, viewer: nerfview.Viewer):
        tabs = server.gui.add_tab_group()
        camera_tab = tabs.add_tab("Rendering Panel", viser.Icon.CAMERA)
        with camera_tab:
            with server.gui.add_folder(label="Set Render Mode", order=1):
                # Camera Mode Dropdown
                camera_type_dropdown = server.gui.add_dropdown(
                    label="Camera Mode",
                    options=["RGB", "Median Depth", "Accumulated Depth", "Rendered Normals", "Normals from Depth"],
                    initial_value="RGB",
                    hint="Select the type of camera."
                )

            @camera_type_dropdown.on_update
            def _handle_camera_type(event):
                keymap = {
                    "RGB": "rgb",
                    "Median Depth": "depth-median",
                    "Accumulated Depth": "depth-acc",
                    "Rendered Normals": "normals-render",
                    "Normals from Depth": "normals-depth"
                }
                self.mode = keymap[camera_type_dropdown.value]
                viewer.rerender(None)

    def _load_gsplat(self, args):
        """
        Load ply file(s) into memory as torch tensors
        """
        torch.manual_seed(42)
        means, quats, scales, colors, opacities = [], [], [], [], []
        for ply_path in args.ply:
            gs_means, gs_quats, gs_scales, gs_colors, gs_opacities = load_ply_data(ply_path)
            means.append(gs_means)
            quats.append(gs_quats)
            scales.append(gs_scales)
            colors.append(gs_colors)
            opacities.append(gs_opacities)

        # Concatenate if multiple PLY files
        self._means = torch.cat(means, dim=0).to(self.device)
        self._quats = torch.cat(quats, dim=0).to(self.device)
        self._scales = torch.cat(scales, dim=0).to(self.device)
        self._opacities = torch.cat(opacities, dim=0).to(self.device)
        self._colors = torch.cat(colors, dim=0).to(self.device)[:,:3]
        self._sh_degree = None
        
        print("Number of Gaussians:", len(self._means))
        print("Using sh degree:", self._sh_degree)
        print_free_gpu_space()

    def render(self, viewmat, K, width, height, get_raw_depth=False):
        """
        Main rendering function for gaussian splats.

        args:
            viewmat: World-to-camera transformation matrix (4x4) torch tensor
            K: Camera intrinsics (3x3) torch tensor
            width: int - width of the image
            height: int - height of the image
        returns:
            ret_img: remdered image, (H,W,3) np.array
        """
        render_width = self.W if self.W else width 
        render_height = self.H if self.H else height

        if int(K[0, -1]) != width // 2 or int(K[1, -1]) != height // 2:
            warnings.warn(
                "Detected mismatch between camera intrinsics and image width and height. "
                "Expected cx == W/2 and cy = H/2. This may cause rendered images to be skewed.",
                UserWarning
            )

        if self.args.backend == "3dgs":
            return rasterization(
                self._means,  # [N, 3]
                self._quats,  # [N, 4]
                self._scales,  # [N, 3]
                self._opacities,  # [N]
                self._colors,  # [N, S, 3]
                viewmat[None],  # [1, 4, 4]
                K[None],  # [1, 3, 3]
                render_width,
                render_height,
                sh_degree=self._sh_degree,
                render_mode="RGB+ED",
                # this is to speedup large-scale rendering by skipping far-away Gaussians.
                radius_clip=3,
            )
        elif self.args.backend == "2dgs":
            return rasterization_2dgs(
                self._means,  # [N, 3]
                self._quats,  # [N, 4]
                self._scales,  # [N, 3]
                self._opacities,  # [N]
                self._colors,  # [N, 3]
                viewmat[None],  # [1, 4, 4]
                K[None],  # [1, 3, 3]
                render_width,
                render_height,
                sh_degree=self._sh_degree,
                render_mode="RGB+ED",
                radius_clip=3,
            )
        else:
            raise ValueError
    
    # register and open viewer
    @torch.no_grad()
    def nerfview_render_fn(self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
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
        2. Selects appropriate rasterization backend (3dgs or 2dgs)
        3. Performs Gaussian splatting using the specified backend
        4. Returns the final rendered image
        """
        width, height = img_wh
        if self.K is not None:
            K = self.K 
        else:
            K = camera_state.get_K(img_wh)
        if self.T is not None:
            viewmat = torch.from_numpy(self.T).float().to(self.device)
        else:
            c2w = camera_state.c2w
            c2w = torch.from_numpy(c2w).float().to(self.device)
            viewmat = c2w.inverse()

        K = torch.from_numpy(K).float().to(self.device)

        if self.stream_T:
            print(viewmat)

        renders = self.render(viewmat, K, width, height)

        if self.args.backend == "3dgs":
            render_colors, render_alphas, meta = renders
        elif self.args.backend == "2dgs":
            (
                render_colors,
                render_alphas,
                render_normals,
                render_normals_from_depth,
                render_distort,
                render_median,
                meta,
            ) = renders
        
        if self.mode == "rgb":
            ret_img = render_colors[0, ..., 0:3].cpu().numpy()

        elif self.mode in ["depth-acc", "depth-median"]:
            if self.mode == "depth-acc":
                render_depths = render_colors[..., 3:4]
            if self.mode == "depth-median":
                render_depths = render_median
            
            min_depth, max_depth = render_depths.min(), render_depths.max()
            render_depths = (((render_depths - min_depth)/ max_depth).cpu().numpy() * 255.0)[0]
            ret_img = np.concatenate(
                [render_depths, render_depths, render_depths],
                axis=-1
            ).astype(np.uint8)

        elif "normal" in self.mode:
            assert self.args.backend == "2dgs", "Surface normals only supported for 2dgs"
            normals = render_normals if self.mode == "normals-render" else render_normals_from_depth
            colored_normals = color_surface_normal(normals, mask=None)
            ret_img = colored_normals.astype(np.uint8)
        else:
            raise NotImplementedError
        
        # Apply fisheye distortion if present
        if self.distortion_params is not None:
            ret_img = distort_image(
                ret_img,
                cam_intr=K.cpu().numpy(),
                dist_coeff=self.distortion_params.reshape((4,)),
            )
        
        return ret_img
    
    # register and open viewer
    @torch.no_grad()
    def simple_render_fn(self, T, K, img_wh: Tuple[int, int], get_raw_depth=False):
        """Render a frame of the 3D Gaussian scene from a given camera viewpoint.

        Args:
            T (np array): World-to-camera transformation matrix
            K (np array): Camera intrinsic parameters
            img_wh (Tuple[int, int]): Target image dimensions (width, height)

        Returns:
            np.ndarray: Rendered RGB image as a numpy array with shape (height, width, 3)
            metadata: Dict of metadata items, such as raw depth data
        """
        width, height = img_wh
        K = torch.from_numpy(K).float().to(self.device)
        viewmat = torch.from_numpy(T).float().to(self.device)
        renders = self.render(viewmat, K, width, height, get_raw_depth=get_raw_depth)
        render_colors = renders[0]
        metadata = {}

        rgb_image = (render_colors[0, ..., 0:3].cpu().numpy() * 255.0).astype(np.uint8)

        render_depths = render_colors[0, ..., 3:4]

        if get_raw_depth:
            metadata["depth_raw"] = render_depths

        # Normalize
        min_depth, max_depth = render_depths.min(), render_depths.max()
        render_depths = (((render_depths - min_depth)/ max_depth).cpu().numpy() * 255.0)
        depth_image = np.concatenate(
            [render_depths, render_depths, render_depths],
            axis=-1
        ).astype(np.uint8)

        # Apply fisheye distortion if present
        if self.distortion_params is not None:
            rgb_image = distort_image(
                rgb_image,
                cam_intr=K.cpu().numpy(),
                dist_coeff=self.distortion_params.reshape((4,)),
            )

            depth_image = distort_image(
                depth_image,
                cam_intr=K.cpu().numpy(),
                dist_coeff=self.distortion_params.reshape((4,)),
            )

        return rgb_image, depth_image, metadata


    def run(self, stream_T=False):
        server = viser.ViserServer(port=self.args.port, verbose=False)
        self.stream_T = stream_T
        viewer = nerfview.Viewer(
            server=server,
            render_fn=self.nerfview_render_fn,
            mode="rendering",
        )
        self._add_gui(server, viewer)
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(100000)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument(
#         "--port", type=int, default=8080, help="port for the viewer server"
#     )
#     parser.add_argument("--backend", type=str, default="3dgs", help="3dgs, 2dgs")
#     parser.add_argument(
#         "--ply", type=str, nargs="+", default=None, help="path to the .ply file(s)", required=True
#     )

#     args = parser.parse_args()

#     assert torch.cuda.is_available(), "Renderer requires a GPU to run. PyTorch cannot detect a GPU."

#     # Simple interactive web viewer
#     # renderer = SimpleRenderer(args)
#     # renderer.run()
    
#     # Or, set custom parameters
#     # Comment the above two lines and uncomment the lines below
#     H, W = 640, 852

#     K = np.array([
#         [403.41072327665165, 0, W/2],
#         [0, 404.223883786868, H/2],
#         [0.0, 0.0, 1.0]
#     ])
    
#     # Fix a position and rotation for the camera. This will disable the interactivity of the web viewer
#     # This T is the world-to-camera transform
#     # Leave T as None to only fix the intrinsics and keep the camera's position and rotation interactive
#     T = np.array([[-8.7352e-01,  4.8679e-01,  5.7073e-09, -4.0487e-01],
#         [ 1.8977e-01,  3.4053e-01, -9.2088e-01,  3.9384e-02],
#         [-4.4828e-01, -8.0441e-01, -3.8984e-01,  1.2369e+00],
#         [ 2.3212e-08, -3.4397e-08, -2.3190e-09,  1.0000e+00]])
    
#     renderer = SimpleRenderer(args, override_K=K, override_T=T, H=H, W=W, distortion_params=D)
#     # To run the web viewer
#     # renderer.run()
#     # To just render an image
#     rgb_im, depth_im, metadata = renderer.simple_render_fn(
#         T, K, (W, H), get_raw_depth=True
#     )
#     imsave("rgb_image.png", rgb_im)
#     imsave("depth_image.png", depth_im)
#     save_depth_as_pcd(metadata["depth_raw"].cpu().numpy(), K, "depth.pcd")