#!/bin/bash
#PJM -L rscgrp=tutorial-share
#PJM -L gpu=1
#PJM -g gt01
#PJM -L jobenv=singularity
#PJM -L elapse=01:00:00
#PJM -j

WORKDIR=/work/01/gt01/GROUP_233-03
IMG_FILE=container.img

cd $WORKDIR
module load singularity
singularity exec --nv --bind $WORKDIR $IMG_FILE


pip uninstall -y causal-conv1d triton  
pip install causal-conv1d==1.2.2.post1 sentencepiece==0.1.99 triton==2.1.0 flask-restful

cd /tmp

git clone https://github.com/state-spaces/mamba.git 
cd mamba 
git checkout v2.0.3 
python setup.py install 
cd .. 
rm -rf mamba

pip install 
