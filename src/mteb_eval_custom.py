import argparse
import mteb
import json
import torch
from mteb.encoder_interface import PromptType
import numpy as np
from models.instract_ir_model import INSRTUCTIRMODEL
from typing import Any
from evaluation.instructions import task_to_instruction
from evaluation.text_formatting_utils import corpus_to_texts
from sentence_transformers import SentenceTransformer


class InstructIRModelWrapper():
    def __init__(self, model=None, task_to_instructions=None):
        self.model = model
        self.tokenizer = self.model.tokenizer

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 128,
        task_name: str,
        prompt_name: str = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.model.encode(sentences, **kwargs, batch_size=batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--peft_model_name_or_path",
        type=str,
        default="../save_model/loss_w_contrastive/checkpoint-1212",
    )
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()

    model = INSRTUCTIRMODEL.from_pretrained(
        args.base_model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )

    model = InstructIRModelWrapper(model=model)
    tasks = mteb.get_tasks(tasks=["Core17InstructionRetrieval", "News21InstructionRetrieval", "Robust04InstructionRetrieval"])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=args.output_dir, batch_size=32)