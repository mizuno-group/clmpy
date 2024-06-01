#!/bin/bash
#PJM -L rscgrp=share
#PJM -L gpu=2
#PJM -g ga97

module load cuda/12.2
module load cudnn/8.9.4
module load cmake/3.24.0
module load gcc/8.3.1

source ~/.bashrc
pyenv local 3.12.0
source ~/torch220env/bin/activate

python3 -m pip install /work/ga97/a97002/clmpy
clmpy.gruvae.train --config experiments/240527/config.yml
clmpy.gruvae.evaluate \
    --config experiments/240527/config.yml \
    --model_path experiments/240527/best_model.pt \
    --test_path data/val_100k.csv