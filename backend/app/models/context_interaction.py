from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


def compute_interaction_components(
    e_c: torch.Tensor, e_p: torch.Tensor
) -> Dict[str, torch.Tensor]:

    d = torch.abs(e_c - e_p)
    m = e_c * e_p
    return {"D": d, "M": m, "concat": torch.cat([e_c, e_p, d, m], dim=-1)}


class ContextInteraction(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        interaction_dim: int = 1024,
        dropout: float = 0.2,
        activation: str = "gelu",
        mode: str = "mlp",
    ):
        super().__init__()
        if mode not in {"mlp", "simple"}:
            raise ValueError(f"Unknown interaction mode: {mode!r}")
        self.hidden_size = int(hidden_size)
        self.interaction_dim = int(interaction_dim)
        self.mode = mode
        self.concat_size = self.hidden_size * (4 if mode == "mlp" else 2)
        self.projection = nn.Linear(self.concat_size, self.interaction_dim)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    @property
    def output_dim(self) -> int:
        return self.interaction_dim

    def forward(
        self, e_c: torch.Tensor, e_p: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if e_p is None:
            e_p = torch.zeros_like(e_c)
        if self.mode == "mlp":
            raw = compute_interaction_components(e_c, e_p)["concat"]
        else:
            raw = torch.cat([e_c, e_p], dim=-1)
        return self.dropout(self.activation(self.projection(raw)))
