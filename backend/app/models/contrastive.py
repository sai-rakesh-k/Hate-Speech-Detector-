
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveProjection(nn.Module):
    """Projects the final representation into the contrastive space (L2-normalised)."""

    def __init__(self, input_dim: int, projection_dim: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(features), dim=-1)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        anchor:    [B, D]        comment-only representation
        positive:  [B, D]        representation with the correct context
        negatives: [B, N, D]     representations with contrasting contexts
        """
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        positive_similarity = (
            torch.sum(anchor * positive, dim=-1, keepdim=True) / self.temperature
        )
        negative_similarities = (
            torch.einsum("bd,bnd->bn", anchor, negatives) / self.temperature
        )
        logits = torch.cat([positive_similarity, negative_similarities], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)
