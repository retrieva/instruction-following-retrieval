import torch
from transformers import (
    AutoModel,
)

class TauModule():
    def __init__(
        self,
        tau_model_name: str = "princeton-nlp/sup-simcse-bert-large-uncased",
        ):
        
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tau_model_name = tau_model_name
        self.tau_model = AutoModel.from_pretrained(tau_model_name).to(device)

    def __call__(
        self,
        tau_sentence_features,
        ):
        
        self.tau_model.eval()
        
        with torch.no_grad():
            q_reps = self.tau_model(**tau_sentence_features[0]).pooler_output # クエリ
            inst_reps_pos = self.tau_model(**tau_sentence_features[1]).pooler_output # 正例指示文
            x_reps_pos = self.tau_model(**tau_sentence_features[2]).pooler_output # クエリ + 正例指示文
            d_reps_pos = self.tau_model(**tau_sentence_features[3]).pooler_output # 正例文書
            d_reps_neg = self.tau_model(**tau_sentence_features[4]).pooler_output # 負例文書

        tau_reps_output = [q_reps, inst_reps_pos, x_reps_pos, d_reps_pos, d_reps_neg]

        return tau_reps_output