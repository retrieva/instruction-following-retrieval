import argparse
import mteb
import json
import torch
from mteb.encoder_interface import PromptType
import numpy as np
from typing import Any
from transformers import AutoModel, AutoTokenizer
import pandas as pd

class LLamaModelWrapper():
    def __init__(self, model_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str = None,
        prompt_name: str = None,
        prompt_type: "PromptType" = None,
        batch_size: int = 8,
        **kwargs,
    ) -> np.ndarray:
        all_embeddings = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_sentences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument("--task_name", type=str, default="STS16")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    model = LLamaModelWrapper(model_name=args.model_name)
    
    tasks = mteb.get_tasks(tasks=["Core17InstructionRetrieval"])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=args.output_dir)