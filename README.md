# simple-renderer
Simple Interface to render a gaussian splat from different views

## Dependencies
First create a virtual environment using the env.yml file. This could be compatible with most conda variants. We like to use `micromamba`[https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html].

```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh) # Install micromamba
micromamba create -f env.yml # Create virtual environment
micromamba activate simple_renderer
pip install nerfview viser tqdm # Install some additional pip dependencies
```

The `gsplat` library is the primary dependency for this library and needs to be built depending on your machine's CUDA version. Some pre-built binaries for different cuda versions are available here[https://docs.gsplat.studio/whl/gsplat/]. The above virtual environment should come with the latest pytorch version (2.4.0)

Install and compile CUDA kernels for gsplat using:
`pip install git+https://github.com/nerfstudio-project/gsplat.git`