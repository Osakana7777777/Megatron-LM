#!/bin/bash
#PJM -L rscgrp=tutorial2-a
#PJM -L elapse=06:00:00
#PJM -g gt01
#PJM -L node=1
#PJM -L jobenv=singularity
#PJM -j 

module load singularity
SIF=/work/gt01/GROUP_233-03/test2.sif
SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY

#singularity exec $SIF echo "start"
#singularity exec --writable-tmpfs --no-home --nv --bind `pwd` --bind /work/gt01/GROUP_233-03/Megatron-LM:/workspace/megatron $SIF python --version
singularity exec --writable-tmpfs --nv --no-home --bind `pwd`  --bind /work/gt01/GROUP_233-03/Megatron-LM:/workspace/megatron $SIF /bin/bash -c "cd /workspace/megatron/scripts/wisteria"  
singularity exec --writable-tmpfs --nv --no-home --bind `pwd`  --bind /work/gt01/GROUP_233-03/Megatron-LM:/workspace/megatron $SIF bash -c " export HOME=/workspace/megatron" 
singularity exec --home /workspace/megatron  --writable-tmpfs --nv --no-home --containall --bind `pwd`  --bind /work/gt01/GROUP_233-03/Megatron-LM:/workspace/megatron $SIF bash -c " echo $HOME"
singularity exec --home /workspace/megatron  --writable-tmpfs --nv --no-home --bind `pwd`   --bind /work/gt01/GROUP_233-03/Megatron-LM:/workspace/megatron $SIF bash -c " wandb login $WANDB_API_KEY"
singularity exec --home /workspace/megatron  --writable-tmpfs --nv --no-home --bind `pwd`  --bind /work/gt01/GROUP_233-03/Megatron-LM:/workspace/megatron $SIF bash train.sh
