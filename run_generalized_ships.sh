#!/bin/sh 
#PJM -g gb20
#PJM -o output.txt
#PJM -e error.txt
#PJM -N ships_test
#PJM -L rscgrp=interactive-a 
#PJM -L node=1
#PJM --at 202501150100

rm -rf miniconda3
module load cuda/12.1
mkdir -p ~/miniconda3/ && cd ~/miniconda3/
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Linux-x86_64.sh && bash Miniconda3-py310_23.10.0-1-Linux-x86_64.sh -b -u -p ~/miniconda3/
source ~/miniconda3/etc/profile.d/conda.sh
cd ..
conda create --name .venv python=3.11 -y
conda install -c nvidia/label/cuda-12.1.0 cuda-toolkit -y
conda install pip -y
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda activate
cd Multilingual-Safety-Head
pip install -r requirements.txt
python3 Run_Generalized_Ships.py