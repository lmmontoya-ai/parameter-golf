from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class BlockAttnResConfig:
    enabled: bool
    block_layers: int


class ParamFreeRMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class BlockAttentionResidual(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.norm = ParamFreeRMSNorm()
        self.query = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))

    def forward(self, completed_blocks: Sequence[Tensor], partial_block: Tensor) -> Tensor:
        candidates = tuple(completed_blocks) + (partial_block,)
        values = torch.stack(candidates, dim=0)
        keys = self.norm(values)
        query = self.query.to(dtype=keys.dtype)
        logits = torch.einsum("d,nbtd->nbt", query, keys)
        weights = logits.softmax(dim=0)
        return torch.einsum("nbt,nbtd->btd", weights, values)
