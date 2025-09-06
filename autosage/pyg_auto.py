import torch
from ._auto import spmm_csr_auto

def _edge_index_to_csr(edge_index: torch.Tensor, num_nodes: int, device=None):
    """
    Convert PyG edge_index [2, E] (src, dst) to CSR for row=dst, col=src.
    """
    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    device = device or edge_index.device
    src = edge_index[0].to(device=device, dtype=torch.long)
    dst = edge_index[1].to(device=device, dtype=torch.long)
    perm = dst.argsort(stable=True)
    dst = dst[perm]; src = src[perm]
    crow = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    crow.index_add_(0, dst + 1, torch.ones_like(dst, dtype=torch.long))
    crow = crow.cumsum(0)
    return crow, src

class AutoSAGEConv(torch.nn.Module):
    """
    Drop-in message-passing-style layer that uses AutoSAGE SpMM for aggregation.
    y[dst] = sum_{(src->dst) in E} x[src]
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, mode: str = "auto"):
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.mode = mode  # placeholder for future

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        num_nodes = x.size(0)
        crow, col = _edge_index_to_csr(edge_index, num_nodes, device=x.device)
        y, info = spmm_csr_auto(crow, col, None if edge_weight is None else edge_weight, x, verbose=False)
        return self.lin(y), info
