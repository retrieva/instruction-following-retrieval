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



l2v = INSRTUCTIRMODEL.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        peft_model_name_or_path="/data/sugiyama/save_model/rankloss/checkpoint-300",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )

instruction = (
    "Given a web search query, retrieve relevant passages that answer the query:"
)
queries = [
    [instruction, "how much protein should a female eat"],
    [instruction, "summit define"],
]
q_reps = l2v.encode(queries)

# Encoding documents. Instruction are not required for documents
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]
d_reps = l2v.encode(documents)

# Compute cosine similarity

q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

print(cos_sim)