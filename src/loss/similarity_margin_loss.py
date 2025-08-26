import torch
from torch import nn, Tensor
from .loss_utils import cos_sim, mismatched_sizes_all_gather


class SimilarityMarginLoss:
    def __init__(
        self,
        similarity_fct=nn.CosineSimilarity(dim=-1),
        alpha: float = 0.4,
        beta: float = 0.1,
    ):
        self.similarity_fct = similarity_fct
        self.alpha = alpha
        self.beta = beta
        self.margin_loss = MarginLoss(alpha=self.alpha, beta=self.beta)

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

            full_d_reps_pos = d_reps_pos # 正例文書
            full_d_reps_neg = d_reps_neg # 指示文を考慮すると関連しない負例文書

            batch_size = d_reps_neg.size(0)
            random_indices = torch.randperm(batch_size, device=d_reps_neg.device)
            full_d_reps_hard_neg = d_reps_neg[random_indices] # バッチ内の無関係な負例文書

        pos_scores = self.similarity_fct(full_x_reps_pos, full_d_reps_pos) # 正例文書：Positive
        neg_scores = self.similarity_fct(full_x_reps_pos, full_d_reps_neg) # 負例文書：Negative
        hard_neg_scores = self.similarity_fct(full_x_reps_pos, full_d_reps_hard_neg) # バッチ内の無関係な負例文書：Hard Negative

        loss = self.margin_loss(neg_scores, pos_scores, hard_neg_scores, similarity_neg_score)
        return loss

class MarginLoss(nn.Module):
    def __init__(self, alpha: float = 0.4, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pos_scores: Tensor, neg_scores: Tensor, hard_neg_scores: Tensor, tau_neg_score: Tensor):
        tau_neg_score = self.alpha * tau_neg_score + self.beta
        print(self.alpha, self.beta)
        loss = torch.clamp(tau_neg_score + neg_scores - pos_scores, min=0.0).sum() + torch.clamp(0.1 + hard_neg_scores - pos_scores, min=0.0).sum()
        return loss