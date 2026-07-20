from __future__ import annotations

import torch


def edge_index_to_csr(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    rows_by: str = "dst",
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if rows_by not in {"dst", "src"}:
        raise ValueError("rows_by must be 'dst' or 'src'")

    source = edge_index[0].to(dtype=torch.long)
    destination = edge_index[1].to(dtype=torch.long)
    row = destination if rows_by == "dst" else source
    col = source if rows_by == "dst" else destination
    order = row.argsort(stable=True)
    row = row[order]
    col = col[order]
    crow = torch.zeros(num_nodes + 1, dtype=torch.long, device=row.device)
    crow.index_add_(0, row + 1, torch.ones_like(row))
    return crow.cumsum(0).contiguous(), col.contiguous()


__all__ = ["edge_index_to_csr"]
