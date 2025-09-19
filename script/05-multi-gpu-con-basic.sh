#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=2
#PJM -L elapse=05:00:00
#PJM -j
#PJM -o ./logs/05-3B-con.txt

module load cuda cudnn nccl gcc
. .venv/bin/activate

SAVE_MODEL="./save_model/05-3B-con"
RESULT_PATH="./results/05-3B-con"

deepspeed ./src/multi-gpu-train.py \
 model.base_model_name="meta-llama/Llama-3.2-3B-Instruct" \
 wandb.name="05-3B-con" \
 wandb.use_wandb=true \
 loss.contrastive_loss=true \
 loss.weighted_contrastive_loss_v1=false \
 loss.weighted_contrastive_loss_v2=false \
 loss.margin_loss=false \
 training.output_dir=${SAVE_MODEL} 