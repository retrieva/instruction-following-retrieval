from dataclasses import dataclass
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from dataset.msmarco import MSMARCO
from models.instract_ir_model import INSRTUCTIRMODEL
from loss.contrastive_loss import ContrastiveLoss
from loss.weighted_contrastive_loss import WeightedContrastiveLoss
from loss.similarity_margin_loss import SimilarityMarginLoss
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import wandb
from distutils.util import strtobool
import argparse

@dataclass
class DefaultCollator:
    model: INSRTUCTIRMODEL
    def __init__(self, model: INSRTUCTIRMODEL) -> None:
        self.model = model

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = features
        num_texts = len(batch[0].texts) # query, d_pos, d_neg, 
        texts = [[] for _ in range(num_texts)]
        labels = []
        similairty_scores = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)
            labels.append(example.label)
            similairty_scores.append(example.similarity_score)
        labels = torch.tensor(labels)
        similairty_scores = torch.tensor(similairty_scores)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.model.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels, similairty_scores

# 訓練をストップさせる
class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class ContrastiveTrainer(Trainer):
    def __init__(
        self,
        *args,
        contrastive_loss=None,
        margin_loss=None,
        weighted_contrastive_loss=None,
        loss_lambda: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.contrastive_loss = contrastive_loss
        self.weighted_contrastive_loss = weighted_contrastive_loss
        self.margin_loss = margin_loss
        self.loss_lambda = loss_lambda

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        
        
        features, labels, tau_score = inputs

        q_reps = self.model(features[0]) # クエリ

        inst_reps_pos = self.model(features[1]) # 正例指示文
        inst_reps_neg = self.model(features[2]) # 負例指示文

        d_reps_pos = self.model(features[1])
        d_reps_neg = self.model(features[2]) 

        x_reps_pos = self.model(features[3])
        x_reps_neg = self.model(features[4]) 

        tau_pos_score = tau_score[:, 0] # 正例の類似度スコア（τ）
        tau_neg_score = tau_score[:, 1] # 負例の類似度スコア（τ）

        if self.contrastive_loss is not None and self.margin_loss is None:
            print("*** Calculate Only Contrastive Loss ... ***")
            #loss = self.contrastive_loss(x_reps_pos, d_reps_pos, d_reps_neg)
            loss = self.weighted_contrastive_loss(q_reps, x_reps_pos, d_reps_pos, d_reps_neg, x_reps_pos, x_reps_neg)

        elif self.contrastive_loss is None and self.margin_loss is not None:
            print("*** Calculate Only Ranking Loss ... ***")
            loss = self.margin_loss(q_reps, d_reps_pos, d_reps_neg, x_reps_pos, tau_neg_score)

        elif self.contrastive_loss is not None and self.margin_loss is not None:
            print("*** Calculate Both Losses ... ***")
            loss = self.contrastive_loss(q_reps, d_reps_pos, d_reps_neg) + self.loss_lambda * self.margin_loss(q_reps, d_reps_pos, d_reps_neg, x_reps_pos, tau_neg_score)

        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
        self.model.save(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def fix_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def initialize_peft(
    model,
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
        "GemmaConfig",
        "Qwen2Config",
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    print(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="InstructIR Training Script")

    parser.add_argument("--similarity_file_path", default="./dataset/similarity/similarity.json")
    # wandb arguments
    parser.add_argument("--wandb_name", default="run-1", help="WandB run name")
    parser.add_argument("--use_wandb", type=lambda x: bool(strtobool(x)), default=False, help="Use wandb logging")

    # save_model_path
    parser.add_argument("--output_dir", default="/data/sugiyama/save_model/contrastive-margin-loss",
                       help="Output directory")
    
    # loss
    parser.add_argument("--use_contrastive_loss", type=lambda x: bool(strtobool(x)), default=False, help="Use contrastive loss")
    parser.add_argument("--use_weighted_contrastive_loss", type=lambda x: bool(strtobool(x)), default=True, help="Use weighted contrastive loss")
    parser.add_argument("--use_margin_loss", type=lambda x: bool(strtobool(x)), default=False, help="Use ranking loss")
    parser.add_argument("--alpha", type=float, default=0.4, help="Alpha parameter for ranking loss")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter for ranking loss")
    parser.add_argument("--loss_lambda", type=float, default=0.1, help="Lambda to balance contrastive and ranking loss")

    return parser.parse_args()


def main():
    args = parse_args()
    fix_seeds()

    msmarco = MSMARCO(similarity_file_path=args.similarity_file_path)
    train_examples = [i for i in msmarco]

    model = INSRTUCTIRMODEL.from_pretrained(
        base_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        enable_bidirectional=True,
    )

    model.model = initialize_peft(
        model.model,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )

    tokenizer = model.tokenizer

    contrastive_loss = ContrastiveLoss() if args.use_contrastive_loss else None
    margin_loss = SimilarityMarginLoss(alpha=args.alpha, beta=args.beta) if args.use_margin_loss else None
    weighted_contrastive_loss = WeightedContrastiveLoss() if args.use_weighted_contrastive_loss else None

    data_collator = DefaultCollator(model)
    
    wandb.init(project="instructir", name=args.wandb_name) if args.use_wandb else None

    training_args = TrainingArguments(
        output_dir = args.output_dir,
        num_train_epochs = 32,
        per_device_train_batch_size = 1,
        logging_steps = 10,
        #gradient_accumulation_steps = 2,
        warmup_steps = 500,
        warmup_ratio = 0.1,
        weight_decay = 0.01,
        learning_rate = 5e-5,
        max_grad_norm = 1.0,
        save_strategy = "steps",
        save_steps = 500,
        fp16=False,
        report_to="wandb" if args.use_wandb else "none",
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        data_collator=data_collator,
        tokenizer=tokenizer,
        contrastive_loss=contrastive_loss,
        margin_loss=margin_loss,
        weighted_contrastive_loss=weighted_contrastive_loss,
        loss_lambda=args.loss_lambda,
    )

    trainer.train()


if __name__ == "__main__":
    main()