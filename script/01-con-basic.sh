#!/bin/sh

SAVE_MODEL="./save_model/01-con"
RESULT_PATH="./results/01-con"

python ./src/train.py \
 wandb.name="01-con" \
 wandb.use_wandb=true \
 loss.contrastive_loss=true \
 loss.weighted_contrastive_loss_v1=false \
 loss.weighted_contrastive_loss_v2=false \
 loss.margin_loss=false \
 training.output_dir=${SAVE_MODEL} 

python ./src/mteb_eval_custom.py \
 --base_model_name_or_path "meta-llama/Llama-3.2-1B-Instruct"\
 --peft_model_name_or_path $(ls -td ${SAVE_MODEL}/checkpoint-* | head -1) \
 --output_dir ${RESULT_PATH}
