import torch
from torch import nn, Tensor
from .loss_utils import cos_sim, mismatched_sizes_all_gather


class ContrastiveLoss:
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
        x_reps: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Tensor = None,
    ):
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        if torch.distributed.is_initialized():
            full_x_reps = mismatched_sizes_all_gather(x_reps)
            full_x_reps = torch.cat(full_x_reps)

            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)
        else:
            full_x_reps = x_reps
            full_d_reps_pos = d_reps_pos
            full_d_reps_neg = d_reps_neg

        full_d_reps = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)
        scores = self.similarity_fct(full_x_reps, full_d_reps) / self.scale

        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )
        loss = self.cross_entropy_loss(scores, labels)

        return loss
    
