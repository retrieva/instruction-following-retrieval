import logging
from dataclasses import dataclass, field
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import seed_worker

from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from dataset.msmarco import MSMARCO
from models.instract_ir_model import INSRTUCTIRMODEL
from loss.contrastive_loss import ContrastiveLoss
from loss.similarity_ranking_loss import SimilarityRankingLoss
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import wandb


@dataclass
class DefaultCollator:
    model: INSRTUCTIRMODEL
    def __init__(self, model: INSRTUCTIRMODEL) -> None:
        self.model = model

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = features
        num_texts = len(batch[0].texts) # query, d_pos, d_neg
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
        ranking_loss=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.contrastive_loss = contrastive_loss
        self.ranking_loss = ranking_loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        
        
        features, labels, similarity_scores = inputs

        q_reps = self.model(features[0]) # 0番目はクエリ

        d_reps_pos = self.model(features[1]) # 1番目は正例文書
        d_reps_neg = self.model(features[2]) # 2番目は指示文を考慮すると関連しない負例文書

        x_reps_pos = self.model(features[3]) # 3番目は、クエリと正例指示文

        similarity_neg_score = similarity_scores[:, 1] # 負例の類似度スコア

        loss1 = self.contrastive_loss(q_reps, d_reps_pos, d_reps_neg)
        loss2 = self.ranking_loss(q_reps, d_reps_pos, d_reps_neg, x_reps_pos, similarity_neg_score)

        loss = loss1 + loss2

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


def main():
    fix_seeds()
    msmarco = MSMARCO()
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

    contrastive_loss = ContrastiveLoss()
    ranking_loss = SimilarityRankingLoss()

    data_collator = DefaultCollator(model)
    
    wandb.init(project="instructir", name="run-1")

    training_args = TrainingArguments(
        output_dir = "/data/sugiyama/save_model/rankloss",
        num_train_epochs = 1,
        per_device_train_batch_size = 8,
        #per_device_eval_batch_size = 32,
        logging_steps = 10,
        gradient_accumulation_steps = 2,
        #warmup_steps = 300,
        warmup_ratio = 0.1,
        weight_decay = 0.01,
        learning_rate = 5e-5,
        max_grad_norm = 1.0,
        #load_best_model_at_end = True,
        save_strategy = "steps",
        save_steps = 200,
        #eval_strategy = "epoch",
        #save_total_limit = 1,
        fp16=False,
        report_to="wandb",
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        data_collator=data_collator,
        tokenizer=tokenizer,
        contrastive_loss=contrastive_loss,
        ranking_loss=ranking_loss,
    )

    trainer.train()


if __name__ == "__main__":
    main()