import torch
from torch import nn, Tensor
from .loss_utils import cos_sim, mismatched_sizes_all_gather

class SimilarityRankingLoss:
    def __init__(
        self,
        scale: float = 20.0,
        similarity_fct=cos_sim,
    ):
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.ranking_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

    def __call__(
        self,
        q_reps: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Tensor = None,
    ):
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        if torch.distributed.is_initialized():
            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)
        else:
            full_q_reps = q_reps
            full_x_reps = x_reps # クエリ+正例指示文

            full_d_reps_pos = d_reps_pos # 正例文書
            full_d_reps_neg = d_reps_neg # 指示文を考慮すると関連しない負例文書

            full_d_reps_hard_neg = d_reps_hard_neg # 無関係な負例文書

        pos_scores = self.similarity_fct(full_x_reps, full_d_reps_pos) # 正例文書：Positive
        neg_scores = self.similarity_fct(full_x_reps, full_d_reps_neg) # 負例文書：Negative
        hard_neg_scores = self.similarity_fct(full_x_reps, full_d_reps_hard_neg) # 無関係な文書：Hard Negative


        loss = self.triplet_loss(neg_scores,pos_scores)
        return loss