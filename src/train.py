from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from dataset.msmarco import MSMARCO
from models.instract_ir_model import INSTRUCTIRMODEL
from loss.contrastive_loss import ContrastiveLoss
from loss.weighted_contrastive_loss import WeightedContrastiveLoss_V1, WeightedContrastiveLoss_V2
from loss.tau_module import TauModule
from loss.margin_loss import MarginLoss
import random
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig
import argparse
import sys
from evaluation.early_stop_callback import SimpleEarlyStoppingCallback

@dataclass
class DefaultCollator:
    model: INSTRUCTIRMODEL
    tau_model_name: str
    def __init__(self, model: INSTRUCTIRMODEL, tau_model_name: str) -> None:
        self.model = model
        self.tau_model_name = tau_model_name
        self.tau_tokenizer = AutoTokenizer.from_pretrained(tau_model_name)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = features
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []


        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)
            labels.append(example.label)
        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.model.tokenize(texts[idx])
            sentence_features.append(tokenized)

        tau_sentence = [example.texts for example in batch]

        return {
            "sentence_features": sentence_features,
            "labels": labels,
            "tau_sentence": tau_sentence
        }

class CustomTrainer(Trainer):
    def __init__(
        self,
        *args,
        contrastive_loss=None,
        margin_loss=None,
        weighted_contrastive_loss_v1=None,
        weighted_contrastive_loss_v2=None,
        tau_classification_module=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.contrastive_loss = contrastive_loss
        self.weighted_contrastive_loss_v1 = weighted_contrastive_loss_v1
        self.weighted_contrastive_loss_v2 = weighted_contrastive_loss_v2
        self.margin_loss = margin_loss
        self.tau_classification_module = tau_classification_module

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        
        
        features = inputs["sentence_features"]
        _ = inputs["labels"]
        tau_sentence = inputs["tau_sentence"]

        q_reps = self.model(features[0]) # クエリ
        inst_reps_pos = self.model(features[1]) # 指示文
        x_reps_pos = self.model(features[2]) # クエリ + 指示文
        d_reps_pos =  self.model(features[3]) # 正例文書
        d_reps_neg =  self.model(features[4]) # 負例文書

        tau_reps_output = None
        if self.margin_loss is not None or self.weighted_contrastive_loss_v1 is not None or self.weighted_contrastive_loss_v2 is not None:
            if self.tau_classification_module is None:
                raise ValueError("Tau module required but not provided.")
            tau_reps_output = self.tau_classification_module(tau_sentence)

        loss_components: List[Tuple[str, torch.Tensor]] = []

        # 対照学習
        if self.contrastive_loss is not None:
            contrastive_loss = self.contrastive_loss(x_reps_pos, d_reps_pos, d_reps_neg)
            loss_components.append(("対照学習損失", contrastive_loss))

        # 重み付き対照学習(手法1)
        if self.weighted_contrastive_loss_v1 is not None:
            if tau_reps_output is None:
                raise ValueError("weighted_contrastive_loss_v1 is set but tau_reps_output is None. Ensure tau_module is configured.")
            weighted_loss_v1 = self.weighted_contrastive_loss_v1(
                q_reps=q_reps,
                inst_reps_pos=inst_reps_pos,
                x_reps_pos=x_reps_pos,
                d_reps_pos=d_reps_pos,
                d_reps_neg=d_reps_neg,
                tau_reps_output=tau_reps_output,
            )
            loss_components.append(("τベース重み付き対照学習損失(手法1)", weighted_loss_v1))
        
        # 重み付き対照学習(手法2)
        if self.weighted_contrastive_loss_v2 is not None:
            if tau_reps_output is None:
                raise ValueError("weighted_contrastive_loss_v2 is set but tau_reps_output is None. Ensure tau_module is configured.")
            weighted_loss_v2 = self.weighted_contrastive_loss_v2(
                q_reps=q_reps,
                inst_reps_pos=inst_reps_pos,
                x_reps_pos=x_reps_pos,
                d_reps_pos=d_reps_pos,
                d_reps_neg=d_reps_neg,
                tau_reps_output=tau_reps_output,
            )
            loss_components.append(("τベース重み付き対照学習損失(手法2)", weighted_loss_v2))
        
        # マージン損失
        if self.margin_loss is not None:
            if tau_reps_output is None:
                raise ValueError("margin_loss is set but tau_reps_output is None. Ensure tau_module is configured.")
            margin_loss = self.margin_loss(x_reps_pos, d_reps_pos, d_reps_neg, tau_reps_output)
            loss_components.append(("マージン損失", margin_loss))

        if not loss_components:
            raise ValueError("No valid loss active. Enable contrastive or weighted_contrastive and/or margin.")

        total_loss = torch.stack([loss for _, loss in loss_components]).sum()
        for label, loss in loss_components:
            print(f"{label}:", loss.item())

        return total_loss
    

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)

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

@hydra.main(config_name="config", version_base=None, config_path="config")
def main(cfg: DictConfig):
    fix_seeds()
    train_dataset = MSMARCO(mode="train")
    dev_dataset = MSMARCO(mode="test")

    model = INSTRUCTIRMODEL.from_pretrained(
        base_model_name_or_path=cfg.model.base_model_name,
        enable_bidirectional=cfg.model.enable_bidirectional
    )

    model.model = initialize_peft(
        model.model,
        lora_r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
    )

    tokenizer = model.tokenizer

    # Loss関数
    contrastive_loss = ContrastiveLoss() if cfg.loss.contrastive_loss else None
    weighted_contrastive_loss_v1 = WeightedContrastiveLoss_V1() if cfg.loss.weighted_contrastive_loss_v1 else None    
    weighted_contrastive_loss_v2 = WeightedContrastiveLoss_V2() if cfg.loss.weighted_contrastive_loss_v2 else None
    margin_loss = MarginLoss() if cfg.loss.margin_loss else None

    tau_classification_module = TauModule(tau_model_name=cfg.model.tau_model_name)
    data_collator = DefaultCollator(model, tau_model_name=cfg.model.tau_model_name)

    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name) if cfg.wandb.use_wandb else None

    training_args = TrainingArguments(
        output_dir = cfg.training.output_dir,
        num_train_epochs = cfg.training.num_train_epochs,
        per_device_train_batch_size = cfg.training.per_device_train_batch_size,
        logging_steps = cfg.training.logging_steps,
        gradient_accumulation_steps = cfg.training.gradient_accumulation_steps,
        warmup_steps = cfg.training.warmup_steps,
        weight_decay = cfg.training.weight_decay,
        learning_rate = cfg.training.learning_rate,
        max_grad_norm = cfg.training.max_grad_norm,
        save_strategy = cfg.training.save_strategy,
        fp16=cfg.training.fp16,
        report_to="wandb" if cfg.wandb.use_wandb else "none",
        evaluation_strategy=cfg.training.eval_strategy,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=False,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
    )


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        contrastive_loss=contrastive_loss,
        margin_loss=margin_loss,
        weighted_contrastive_loss_v1=weighted_contrastive_loss_v1,
        weighted_contrastive_loss_v2=weighted_contrastive_loss_v2,
        tau_classification_module=tau_classification_module,
        callbacks=[
            SimpleEarlyStoppingCallback(
                metric_name="eval_loss",
                greater_is_better=False,
                patience=1,
                min_delta=0.0,
            )
        ]
    )

    trainer.train()


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--local_rank", type=int, default=-1)
    known, unknown = pre_parser.parse_known_args()

    if known.local_rank != -1:
        os.environ["LOCAL_RANK"] = str(known.local_rank)
    sys.argv = [sys.argv[0]] + unknown

    main()
