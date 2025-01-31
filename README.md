# Simple Renderer
Simple interface to render a gaussian splat from different views

## Dependencies
We use [`micromamba`](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) to create a virtual environment.

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

Examples scripts can be found under `examples`

# Web Viewer
To launch a simple web viewer, run

```bash
source setup.sh
python examples/web_viewer.py # loads default lucky cat asset
python examples/web_viewer.py --ply <PATH> --backend "[2dgs or 3dgs]" # Renders your gaussian splat
```

# Rendering without interactive viewer
You can also simply render RGBD data without a web viewer. See the script `examples/render_images.py` for an example.

This script will save 3 outputs:
    - RGB Image
    - Depth Image
    - Depth Point Cloud

# Fisheye Cameras
The above functions can also be used for fisheye cameras. See `examples/fisheye_camera.py` for an example.

The major difference to note is that you must provide the distortion coefficients as a 4-dim np array.