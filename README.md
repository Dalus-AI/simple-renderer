# simple-renderer
Simple Interface to render a gaussian splat from different views

## Dependencies
We use `micromamba`[https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html] to create a virtual environment.

To install micromamba, run:
```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh) # Install micromamba
```

Then, run the bash script for installation:

```bash
chmod +x install.sh
bash install.sh # This may take some time
```

The `gsplat` library is the primary dependency for this library and needs to be built depending on your machine's CUDA version. The above command will install all dependencies and build the CUDA kernels for gsplat. 

To render a PLY file, you can then simply use:

```bash
micromamba activate simple_renderer
python render.py --ply <PATH_TO_PLY_FILE> --web_viewer
python render.py --ply <PATH_TO_PLY_FILE1> --ply <PATH_TO_PLY_FILE2> --web_viewer # Render a scene with multiple PLY files
```

Then navigate to the URL that is printed on screen to view the live rendering.

The script also has a function `simple_render_fn(T, K, img_wh: Tuple[int, int])` that can render images without a web viewer.

It takes the following inputs:
- T: np.array (4,4) representing world-to-camera Transformation Matrix (Camera extrinsics)
- K: np.array (3,3) representing the camera's intrinsics parameter matrix
- img_wh: tuple(int, int) representing the width and height of the output image needed

It will return a (w,h,3) matrix representing the RGB image from the camera position.