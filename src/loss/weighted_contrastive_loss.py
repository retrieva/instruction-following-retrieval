import torch
from torch import nn, Tensor
from .loss_utils import cos_sim, mismatched_sizes_all_gather, tau_weighted
from typing import List
import torch.nn.functional as F


def multi_positive_weighted_loss(logits, weight):
    log_probs = F.log_softmax(logits, dim=1)
    weight_norm = weight / (weight.sum(dim=1, keepdim=True))
    loss = -(weight_norm * log_probs).sum(dim=1).mean()
    
    return loss


class WeightedContrastiveLoss_V1:
    def __init__(
        self,
        scale: float = 0.05,
        similarity_fct=cos_sim,
    ):
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def __call__(
        self,
        q_reps: Tensor,
        inst_reps_pos: Tensor,
        x_reps_pos: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Tensor,
        tau_reps_output: List[Tensor],
    ):
        
        if torch.distributed.is_initialized():
            # 訓練モデルの埋め込み
            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_inst_reps_pos = mismatched_sizes_all_gather(inst_reps_pos)
            full_inst_reps_pos = torch.cat(full_inst_reps_pos)

            full_x_reps_pos = mismatched_sizes_all_gather(x_reps_pos)
            full_x_reps_pos = torch.cat(full_x_reps_pos)

            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)

            # τを計算したモデルの埋め込み
            full_tau_q_reps = mismatched_sizes_all_gather(tau_reps_output[0])
            full_tau_q_reps = torch.cat(full_tau_q_reps)

            full_tau_inst_reps_pos = mismatched_sizes_all_gather(tau_reps_output[1])
            full_tau_inst_reps_pos = torch.cat(full_tau_inst_reps_pos)

            full_tau_x_reps_pos = mismatched_sizes_all_gather(tau_reps_output[2])
            full_tau_x_reps_pos = torch.cat(full_tau_x_reps_pos)

            full_tau_d_reps_pos = mismatched_sizes_all_gather(tau_reps_output[3])
            full_tau_d_reps_pos = torch.cat(full_tau_d_reps_pos)

            full_tau_d_reps_neg = mismatched_sizes_all_gather(tau_reps_output[4])
            full_tau_d_reps_neg = torch.cat(full_tau_d_reps_neg)

        else:
            full_q_reps = q_reps

            full_inst_reps_pos = inst_reps_pos

            full_x_reps_pos = x_reps_pos

            full_d_reps_pos = d_reps_pos

            full_d_reps_neg = d_reps_neg

            full_tau_q_reps = tau_reps_output[0]

            full_tau_inst_reps_pos = tau_reps_output[1]

            full_tau_x_reps_pos = tau_reps_output[2]

            full_tau_d_reps_pos = tau_reps_output[3]

        # 重みを計算する
        tau_q_d_scores = self.similarity_fct(full_tau_q_reps, full_tau_d_reps_pos) / self.scale
        tau_q_weights = torch.diag(tau_q_d_scores)
        tau_q_weights = torch.diag(tau_q_weights)

        tau_inst_pos_d_scores = self.similarity_fct(full_tau_inst_reps_pos, full_tau_d_reps_pos) / self.scale
        tau_inst_weights = torch.diag(tau_inst_pos_d_scores+0.1)
        tau_inst_weights = torch.diag(tau_inst_weights)

        tau_x_pos_d_scores = self.similarity_fct(full_tau_x_reps_pos, full_tau_d_reps_pos) / self.scale
        tau_x_weights = torch.diag(tau_x_pos_d_scores+0.2)
        tau_x_weights = torch.diag(tau_x_weights)

        # 損失計算
        full_q_inst_x_reps = torch.cat([full_q_reps, full_inst_reps_pos, full_x_reps_pos], dim=0)
        full_d_reps_pos_score = self.similarity_fct(full_d_reps_pos, full_q_inst_x_reps) / self.scale

        weighted = torch.cat([tau_q_weights, tau_inst_weights, tau_x_weights], dim=1)

        loss = multi_positive_weighted_loss(full_d_reps_pos_score, weighted)

        return loss
    
class WeightedContrastiveLoss_V2:
    def __init__(
        self,
        scale: float = 0.05,
        similarity_fct=cos_sim,
    ):
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def __call__(
        self,
        q_reps: Tensor,
        inst_reps_pos: Tensor,
        x_reps_pos: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Tensor,
        tau_reps_output: List[Tensor],
    ):
        
        if torch.distributed.is_initialized():
            # 訓練モデルの埋め込み
            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_inst_reps_pos = mismatched_sizes_all_gather(inst_reps_pos)
            full_inst_reps_pos = torch.cat(full_inst_reps_pos)

            full_x_reps_pos = mismatched_sizes_all_gather(x_reps_pos)
            full_x_reps_pos = torch.cat(full_x_reps_pos)

            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)

            # τを計算したモデルの埋め込み
            full_tau_q_reps = mismatched_sizes_all_gather(tau_reps_output[0])
            full_tau_q_reps = torch.cat(full_tau_q_reps)

            full_tau_inst_reps_pos = mismatched_sizes_all_gather(tau_reps_output[1])
            full_tau_inst_reps_pos = torch.cat(full_tau_inst_reps_pos)

            full_tau_x_reps_pos = mismatched_sizes_all_gather(tau_reps_output[2])
            full_tau_x_reps_pos = torch.cat(full_tau_x_reps_pos)

            full_tau_d_reps_pos = mismatched_sizes_all_gather(tau_reps_output[3])
            full_tau_d_reps_pos = torch.cat(full_tau_d_reps_pos)

            full_tau_d_reps_neg = mismatched_sizes_all_gather(tau_reps_output[4])
            full_tau_d_reps_neg = torch.cat(full_tau_d_reps_neg)

        else:
            full_q_reps = q_reps

            full_inst_reps_pos = inst_reps_pos

            full_x_reps_pos = x_reps_pos

            full_d_reps_pos = d_reps_pos

            full_d_reps_neg = d_reps_neg

            full_tau_q_reps = tau_reps_output[0]

            full_tau_inst_reps_pos = tau_reps_output[1]

            full_tau_x_reps_pos = tau_reps_output[2]

            full_tau_d_reps_pos = tau_reps_output[3]

            full_tau_d_reps_neg = tau_reps_output[4]

        full_d_reps = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)

        q_scores = self.similarity_fct(full_q_reps, full_d_reps) / self.scale

        inst_scores = self.similarity_fct(full_inst_reps_pos, full_d_reps) / self.scale

        x_scores = self.similarity_fct(full_x_reps_pos, full_d_reps) / self.scale

        full_tau_d_reps = torch.cat([full_tau_d_reps_pos, full_tau_d_reps_neg], dim=0)

        # τ：[クエリ, 正例文書+負例文書]
        tau_q_scores = self.similarity_fct(full_tau_q_reps, full_tau_d_reps) / self.scale
        num_pos_docs = tau_q_scores.size(0)
        tau_q_weighted = tau_weighted(tau_q_scores)

        # τ：[指示文, 正例文書+負例文書]
        tau_inst_pos_scores = self.similarity_fct(full_tau_inst_reps_pos, full_tau_d_reps)
        num_pos_docs = tau_inst_pos_scores.size(0)
        for i in range(num_pos_docs):
            tau_inst_pos_scores[i, i] += 0.1
        tau_inst_pos_scores = tau_inst_pos_scores / self.scale
        tau_inst_pos_weighted = tau_weighted(tau_inst_pos_scores)

        # τ：[クエリ+指示文, 正例文書+負例文書]
        tau_x_pos_scores = self.similarity_fct(full_tau_x_reps_pos, full_tau_d_reps)
        num_pos_docs = tau_x_pos_scores.size(0)
        for i in range(num_pos_docs):
            tau_x_pos_scores[i, i] += 0.2
        tau_x_pos_scores = tau_x_pos_scores / self.scale
        tau_x_pos_weighted = tau_weighted(tau_x_pos_scores)

        weighted = torch.stack([tau_q_weighted, tau_inst_pos_weighted, tau_x_pos_weighted], dim=0) # (3, batch_size)
        normalize_weighted = weighted.view(-1) # (3*batch_size)

        # 実際の損失計算
        labels = torch.tensor(
            range(len(full_q_reps)), dtype=torch.long, device=full_d_reps.device
        )
        labels = torch.cat([labels, labels, labels]) 
        losses = self.cross_entropy_loss(
            torch.cat([q_scores, inst_scores, x_scores], dim=0),
            labels
        )

        loss = torch.mean(losses * normalize_weighted)

        return loss