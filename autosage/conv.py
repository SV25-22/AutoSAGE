from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ._auto import spmm_csr_auto
from .ops import spmm_csr
from .utils.csr import edge_index_to_csr


class AutoSAGEConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        mode: str = "auto",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if mode not in {"auto", "native"}:
            raise ValueError("mode must be 'auto' or 'native'")
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.mode = mode

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        *,
        return_info: bool = False,
    ):
        crow, col = edge_index_to_csr(edge_index, x.size(0), rows_by="dst")
        if edge_weight is not None:
            if edge_weight.ndim != 1 or edge_weight.numel() != edge_index.size(1):
                raise ValueError("edge_weight must contain one value per edge")
            if edge_weight.device != edge_index.device:
                raise ValueError(
                    "edge_weight and edge_index must be on the same device"
                )
            order = edge_index[1].to(dtype=torch.long).argsort(stable=True)
            edge_weight = edge_weight.index_select(0, order)
        if self.mode == "auto":
            aggregated, info = spmm_csr_auto(crow, col, edge_weight, x)
        else:
            aggregated = spmm_csr(crow, col, edge_weight, x)
            info = {
                "operation": "spmm",
                "choice": "native" if x.is_cuda else "baseline",
                "from_cache": False,
            }
        output = self.linear(aggregated)
        return (output, info) if return_info else output
