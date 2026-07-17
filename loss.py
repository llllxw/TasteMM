import torch
import torch.nn as nn


class SupConHardLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        if batch_size < 2:
            return features.sum() * 0.0

        positive_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(features.device)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        positive_mask = positive_mask.masked_fill(self_mask, 0)

        similarity = torch.matmul(features, features.t()) / self.temperature
        negative_mask = ~(positive_mask.bool() | self_mask)
        hard_negative_mask = torch.zeros_like(similarity, device=features.device)
        for idx in range(batch_size):
            negative_indices = torch.where(negative_mask[idx])[0]
            if negative_indices.numel() == 0:
                continue
            k = min(3, int(negative_indices.numel()))
            candidate_similarity = similarity[idx, negative_indices]
            selected = negative_indices[torch.topk(candidate_similarity, k=k).indices]
            hard_negative_mask[idx, selected] = 1

        exp_similarity = torch.exp(similarity - similarity.max(dim=1, keepdim=True).values.detach())
        positive_exp = (exp_similarity * positive_mask).sum(dim=1) + 1e-10
        negative_exp = (exp_similarity * hard_negative_mask).sum(dim=1) + 1e-10
        loss = -torch.log(positive_exp / (positive_exp + negative_exp))
        valid_anchors = (positive_mask.sum(dim=1) > 0) & (negative_mask.sum(dim=1) > 0)
        if not torch.any(valid_anchors):
            return features.sum() * 0.0
        return loss[valid_anchors].mean()
