import argparse
import mteb
import torch
from mteb.encoder_interface import PromptType
import numpy as np
from models.instract_ir_model import INSTRUCTIRMODEL
from typing import Any


class InstructIRModelWrapper():
    def __init__(self, model=None, task_to_instructions=None):
        #self.task_to_instructions = task_to_instructions
        self.model = model

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_name: str = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
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
    )
    parser.add_argument("--output_dir", type=str, default="results/")

    args = parser.parse_args()

    model = INSTRUCTIRMODEL.from_pretrained(
        args.base_model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )

    model = InstructIRModelWrapper(model=model)
    tasks = mteb.get_tasks(tasks=["Core17InstructionRetrieval", "News21InstructionRetrieval", "Robust04InstructionRetrieval", "STS12", "STS13", "STS14", "STS15", "STS16"])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=args.output_dir)


