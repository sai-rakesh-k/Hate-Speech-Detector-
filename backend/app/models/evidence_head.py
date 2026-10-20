from __future__ import annotations

import torch
import torch.nn as nn


class EvidenceHead(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(token_embeddings).squeeze(-1)

    @torch.no_grad()
    def token_probabilities(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Sigmoid probabilities per token (``[B, T]``)."""
        return torch.sigmoid(self.classifier(token_embeddings).squeeze(-1))
