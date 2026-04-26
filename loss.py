import torch
import torch.nn as nn


class SupConHardLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        positive_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(features.device)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        positive_mask = positive_mask.masked_fill(self_mask, 0)

        similarity = torch.matmul(features, features.t()) / self.temperature
        negative_similarity = similarity.masked_fill(positive_mask.bool(), -1e9)
        _, hard_negative_indices = torch.topk(negative_similarity, k=3, dim=1)

        hard_negative_mask = torch.zeros_like(similarity, device=features.device)
        for idx in range(batch_size):
            hard_negative_mask[idx, hard_negative_indices[idx]] = 1

        exp_similarity = torch.exp(similarity)
        positive_exp = (exp_similarity * positive_mask).sum(dim=1) + 1e-10
        negative_exp = (exp_similarity * hard_negative_mask).sum(dim=1) + 1e-10
        loss = -torch.log(positive_exp / (positive_exp + negative_exp))
        return loss.mean()
