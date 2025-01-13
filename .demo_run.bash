#!/bin/sh 
#PJM -g gb20
#PJM -o output.txt
#PJM -e error.txt
#PJM -N Generalized_Ships
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM --at 202501140015

mkdir ./miniconda3/ && cd ~/miniconda3/
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Linux-x86_64.sh && bash Miniconda3-py310_23.10.0-1-Linux-x86_64.sh -b -u -p ~/miniconda3/
source ./miniconda3/etc/profile.d/conda.sh
which conda && echo "====" && conda --version
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.2 -c pytorch -c nvidia

conda activate
cd ..
cd Multilingual-Safety-Head
pip3 install requirements.txt
python3 Run_Generalized_Ships.py