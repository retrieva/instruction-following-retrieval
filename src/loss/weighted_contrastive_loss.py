import torch
from torch import nn, Tensor
from .loss_utils import cos_sim, mismatched_sizes_all_gather


class WeightedContrastiveLoss:
    def __init__(
        self,
        scale: float = 0.05,
        similarity_fct=cos_sim,
    ):
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(
        self,
        q_reps: Tensor,
        inst_reps_pos: Tensor,
        inst_reps_neg: Tensor,
        x_reps_pos: Tensor,
        x_reps_neg: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Tensor,
        tau_q_and_p_pos_score
        tau_q_and_p_neg_score

        tau_inst_pos_and_p_pos_score
        tau_inst_pos_and_p_neg_score = tau_score[:, 3]  # 正例指示文と負例文書スコア（τ）

        tau_inst_neg_and_p_pos_score = tau_score[:, 4]  # 負例指示文と正例文書スコア（τ）
        tau_inst_neg_and_p_neg_score = tau_score[:, 5]  # 負例指示文と負例文書スコア（τ）

        tau_x_pos_and_p_pos_score = tau_score[:, 6]  # クエリ+正例指示文と正例文書スコア（τ）
        tau_x_pos_and_p_neg_score = tau_score[:, 7]  # クエリ+正例指示文と負例文書スコア（τ）
        
        tau_x_neg_and_p_pos_score = tau_score[:, 8]  # クエリ+負例指示文と正例文書スコア（τ）
        tau_x_neg_and_p_neg_score = tau_score[:, 9]  # クエリ+負例指示文と負例文書スコア（τ）



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
            full_d_reps_pos = d_reps_pos # 正例文書
            full_q_reps = q_reps # クエリと指示文
            full_d_reps_neg = d_reps_neg # 負例文書

        d_reps = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)
        scores = self.similarity_fct(full_q_reps, d_reps) / self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )

        loss = self.cross_entropy_loss(scores, labels)
        return loss
    
