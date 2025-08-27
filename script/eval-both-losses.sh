#!/bin/sh
## PJM -L rscgrp=b-batch
## PJM -L gpu=1
## PJM -L elapse=00:35:00
## PJM -j

# module load cuda cudnn nccl gcc
# . .venv/bin/activate

SAVEMODEL_BASE_PATH="./save_model"
RESULTM_BASE_PATH="./src/results"

CONTRASTIVE_LOSS=True
MARGIN_LOSS=True
ALPHA=0.4
BETA=0.1


if [ "$CONTRASTIVE_LOSS" = "True" ] && [ "$MARGIN_LOSS" = "False" ]; then
    FOLDER_NAME="loss_w_contrastive"
elif [ "$CONTRASTIVE_LOSS" = "False" ] && [ "$MARGIN_LOSS" = "True" ]; then
    FOLDER_NAME="loss_w_margin_alpha${ALPHA}_beta${BETA}"
else
    FOLDER_NAME="loss_w_contrastive_margin_alpha${ALPHA}_beta${BETA}"
fi

SAVEMODEL_PATH="${SAVEMODEL_BASE_PATH}/${FOLDER_NAME}"
RESULT_PATH="${RESULTM_BASE_PATH}/${FOLDER_NAME}"

python ./src/mteb_eval_custom.py \
 --base_model_name_or_path meta-llama/Llama-3.2-1B-Instruct\
 --peft_model_name_or_path ${SAVEMODEL_PATH} \
 --output_dir ${RESULT_PATH}