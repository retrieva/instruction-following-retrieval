#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=00:35:00
#PJM -j

module load cuda cudnn nccl gcc
. .venv/bin/activate

BASE_PATH="./save_model"

USE_WANDB=False
CONTRASTIVE_LOSS=True
MARGIN_LOSS=False
ALPHA=0.4
BETA=0.1


if [ "$CONTRASTIVE_LOSS" = "True" ] && [ "$MARGIN_LOSS" = "False" ]; then
    FOLDER_NAME="loss_w_contrastive"
    WANDB_NAME="loss_w_contrastive"
elif [ "$CONTRASTIVE_LOSS" = "False" ] && [ "$MARGIN_LOSS" = "True" ]; then
    FOLDER_NAME="loss_w_margin_alpha${ALPHA}_beta${BETA}"
    WANDB_NAME="loss_w_margin_alpha${ALPHA}_beta${BETA}"
else
    FOLDER_NAME="loss_w_contrastive_margin_alpha${ALPHA}_beta${BETA}"
    WANDB_NAME="loss_w_contrastive_margin_alpha${ALPHA}_beta${BETA}"
fi

FULL_PATH="${BASE_PATH}/${FOLDER_NAME}"
mkdir -p ${FULL_PATH}

python ./src/train.py \
 --similarity_file_path ./dataset/similarity/similarity.json \
 --wandb_name ${WANDB_NAME} \
 --use_wandb ${USE_WANDB} \
 --output_dir ${FULL_PATH} \
 --use_contrastive_loss ${CONTRASTIVE_LOSS} \
 --use_margin_loss ${MARGIN_LOSS} \
 --alpha ${ALPHA} \
 --beta ${BETA}