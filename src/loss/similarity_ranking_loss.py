import torch
from torch import nn, Tensor
from .loss_utils import cos_sim, mismatched_sizes_all_gather

class SimilarityRankingLoss:
    def __init__(
        self,
        similarity_fct=nn.CosineSimilarity(dim=-1),
    ):
        self.similarity_fct = similarity_fct
        self.triplet_loss = MarginRankingLoss()

    def __call__(
        self,
        q_reps: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Tensor,
        x_reps_pos: Tensor,
        similarity_neg_score: Tensor = None,
    ):

        if torch.distributed.is_initialized():
            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_x_reps_pos = mismatched_sizes_all_gather(x_reps_pos)
            full_x_reps_pos = torch.cat(full_x_reps_pos)

            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)
        else:
            full_q_reps = q_reps
            full_x_reps_pos = x_reps_pos # クエリと正例指示文
            #full_x_reps_neg = x_reps_neg # クエリと負例指示文

            full_d_reps_pos = d_reps_pos # 正例文書
            full_d_reps_neg = d_reps_neg # 指示文を考慮すると関連しない負例文書
        
        pos_scores = self.similarity_fct(full_x_reps_pos, full_d_reps_pos) # 正例文書：Positive
        neg_scores = self.similarity_fct(full_x_reps_pos, full_d_reps_neg) # 負例文書：Negative

        loss = self.triplet_loss(neg_scores, pos_scores, similarity_neg_score)

        return loss

class MarginRankingLoss(nn.Module):
    def __init__(self, alpha: float = 0.4, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pos_scores: Tensor, neg_scores: Tensor, similarity_neg_score: Tensor):
        similarity_neg_score = self.alpha * similarity_neg_score + self.beta
        return torch.clamp(similarity_neg_score - pos_scores + neg_scores, min=0.0).mean()