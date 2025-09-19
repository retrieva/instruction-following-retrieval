#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=04:00:00
#PJM -j
#PJM -o ./logs/03-weight-v2.txt

module load cuda cudnn nccl gcc
. .venv/bin/activate

SAVE_MODEL="./save_model/03-weight-v2"
RESULT_PATH="./results/03-weight-v2"

python ./src/train.py \
 wandb.name="03-weight-v2" \
 wandb.use_wandb=true \
 loss.contrastive_loss=false \
 loss.weighted_contrastive_loss_v1=false \
 loss.weighted_contrastive_loss_v2=true \
 loss.margin_loss=false \
 training.output_dir=${SAVE_MODEL} 

python ./src/mteb_eval_custom.py \
 --base_model_name_or_path "meta-llama/Llama-3.2-1B-Instruct"\
 --peft_model_name_or_path $(ls -td ${SAVE_MODEL}/checkpoint-* | head -1) \
 --output_dir ${RESULT_PATH}
