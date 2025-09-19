import torch
from torch import nn, Tensor
from .loss_utils import mismatched_sizes_all_gather
from typing import List

class MarginLoss:
    def __init__(
        self,
        similarity_fct=nn.CosineSimilarity(dim=-1),
    ):
        self.similarity_fct = similarity_fct
        self.margin_loss = MarginLossFunction()

    def __call__(
        self,
        x_reps_pos: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Tensor,
        tau_reps_output: List[Tensor],
    ):

        if torch.distributed.is_initialized():
            x_reps_pos = mismatched_sizes_all_gather(x_reps_pos)
            x_reps_pos = torch.cat(x_reps_pos)
        
            d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            d_reps_pos = torch.cat(d_reps_pos)

            d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            d_reps_neg = torch.cat(d_reps_neg)
        else:
            x_reps_pos = x_reps_pos # クエリと正例指示文

            d_reps_pos = d_reps_pos # 正例文書
            d_reps_neg = d_reps_neg # 指示文を考慮すると関連しない負例文書

        pos_scores = self.similarity_fct(x_reps_pos, d_reps_pos)
        neg_scores = self.similarity_fct(x_reps_pos, d_reps_neg)

        # クエリ+指示文/負例文書
        x_reps = tau_reps_output[2]
        d_neg_reps = tau_reps_output[4]
        margin = self.similarity_fct(x_reps, d_neg_reps)
        loss = self.margin_loss(pos_scores, neg_scores, margin)

        return loss


class MarginLossFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_scores: Tensor, neg_scores: Tensor, margin: Tensor):
        margin = 0.4 * margin + 0.1
        loss = torch.clamp(neg_scores - pos_scores + margin, min=0.0).mean()
        return loss
