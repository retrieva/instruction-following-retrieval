#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=05:00:00
#PJM -j
#PJM -o ./logs/04-margin.txt

module load cuda cudnn nccl gcc
. .venv/bin/activate

SAVE_MODEL="./save_model/05-margin"
RESULT_PATH="./results/05-margin"

deepspeed ./src/train.py \
 wandb.name="04-margin" \
 wandb.use_wandb=true \
 loss.contrastive_loss=false \
 loss.weighted_contrastive_loss_v1=false \
 loss.weighted_contrastive_loss_v2=false \
 loss.margin_loss=true \
 training.output_dir=${SAVE_MODEL} 