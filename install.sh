#!/bin/bash

eval "$(micromamba shell hook --shell bash)"

echo "Creating base environment"
micromamba create -n simple_renderer python=3.10 -c conda -c conda-forge -y
micromamba activate simple_renderer

echo "Installing pytorch and basic dependencies"
micromamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install nerfview viser tqdm # Install some additional pip dependencies

echo "Installing and Building gsplat. This may take some time"
pip install git+https://github.com/nerfstudio-project/gsplat.git
