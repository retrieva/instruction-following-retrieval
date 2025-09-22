#!/bin/sh

# 対照学習
SAVE_MODEL="/data/sugiyama/save_models_on_experiments/01-con"
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


# 重み付き対照学習v1
SAVE_MODEL="/data/sugiyama/save_models_on_experiments/02-weight-v1"
RESULT_PATH="./results/02-weight-v1"

python ./src/train.py \
 wandb.name="02-weight-v1" \
 wandb.use_wandb=true \
 loss.contrastive_loss=true \
 loss.weighted_contrastive_loss_v1=true \
 loss.weighted_contrastive_loss_v2=false \
 loss.margin_loss=false \
 training.output_dir=${SAVE_MODEL} 

python ./src/mteb_eval_custom.py \
 --base_model_name_or_path "meta-llama/Llama-3.2-1B-Instruct"\
 --peft_model_name_or_path $(ls -td ${SAVE_MODEL}/checkpoint-* | head -1) \
 --output_dir ${RESULT_PATH}


# 重み付き対照学習v2

SAVE_MODEL="/data/sugiyama/save_models_on_experiments/03-weight-v2"
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

# マージン付き対照学習

SAVE_MODEL="/data/sugiyama/save_models_on_experiments/04-margin"
RESULT_PATH="./results/04-margin"

python ./src/train.py \
 wandb.name="04-margin" \
 wandb.use_wandb=true \
 loss.contrastive_loss=false \
 loss.weighted_contrastive_loss_v1=false \
 loss.weighted_contrastive_loss_v2=false \
 loss.margin_loss=true \
 training.output_dir=${SAVE_MODEL} 

python ./src/mteb_eval_custom.py \
 --base_model_name_or_path "meta-llama/Llama-3.2-1B-Instruct"\
 --peft_model_name_or_path $(ls -td ${SAVE_MODEL}/checkpoint-* | head -1) \
 --output_dir ${RESULT_PATH}

# 対照学習+マージン付き対照学習
SAVE_MODEL="/data/sugiyama/save_models_on_experiments/05-con_basic_margin"
RESULT_PATH="./results/05-con_basic_margin"

python ./src/train.py \
 wandb.name="05-con_basic_margin" \
 wandb.use_wandb=true \
 loss.contrastive_loss=true \
 loss.weighted_contrastive_loss_v1=false \
 loss.weighted_contrastive_loss_v2=false \
 loss.margin_loss=true \
 training.output_dir=${SAVE_MODEL} 

python ./src/mteb_eval_custom.py \
 --base_model_name_or_path "meta-llama/Llama-3.2-1B-Instruct"\
 --peft_model_name_or_path $(ls -td ${SAVE_MODEL}/checkpoint-* | head -1) \
 --output_dir ${RESULT_PATH}


# 重み付き対照学習+マージン付き対照学習
SAVE_MODEL="/data/sugiyama/save_models_on_experiments/06-weight_margin"
RESULT_PATH="./results/06-weight_margin"

python ./src/train.py \
 wandb.name="06-weight_margin" \
 wandb.use_wandb=true \
 loss.contrastive_loss=false \
 loss.weighted_contrastive_loss_v1=false \
 loss.weighted_contrastive_loss_v2=true \
 loss.margin_loss=true \
 training.output_dir=${SAVE_MODEL} 

python ./src/mteb_eval_custom.py \
 --base_model_name_or_path "meta-llama/Llama-3.2-1B-Instruct"\
 --peft_model_name_or_path $(ls -td ${SAVE_MODEL}/checkpoint-* | head -1) \
 --output_dir ${RESULT_PATH}