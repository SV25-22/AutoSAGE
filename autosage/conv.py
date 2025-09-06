import torch
from torch import nn
from .ops import spmm_csr

class AutoSAGEConv(nn.Module):
    """
    P0 stub: behaves like sum aggregator Y = A * X
    (mode='auto' is accepted but not used yet; scheduler lands later)
    """
    def __init__(self, in_ch: int, out_ch: int, mode: str = "auto", bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch, bias=bias)
        self.mode = mode

    def forward(self, x: torch.Tensor, crow: torch.Tensor, col: torch.Tensor, edge_weight=None):
        # 1) aggregate neighbors
        agg = spmm_csr(crow, col, edge_weight, x)
        # 2) project
        return self.lin(agg)
