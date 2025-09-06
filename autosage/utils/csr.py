import torch

def edge_index_to_csr(edge_index: torch.Tensor, num_nodes: int):
    """
    edge_index: [2, E] (row -> col) COO
    Returns: crow [N+1], col [E] as int64
    """
    row, col = edge_index[0], edge_index[1]
    perm = row.argsort(stable=True)
    row, col = row[perm], col[perm]
    crow = torch.zeros(num_nodes + 1, dtype=torch.long, device=row.device)
    crow.index_add_(0, row + 1, torch.ones_like(row, dtype=torch.long))
    crow = crow.cumsum(0)
    return crow.to(torch.long).contiguous(), col.to(torch.long).contiguous()
