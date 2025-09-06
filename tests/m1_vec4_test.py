#!/usr/bin/env python3
import os, torch, random
torch.ops.load_library("cmake-build/libautosage_cuda.so")

def rand_csr(N, avg_deg):
    import numpy as np
    M = int(N*avg_deg)
    row = np.random.randint(0, N, size=M, dtype=np.int64)
    col = np.random.randint(0, N, size=M, dtype=np.int64)
    pair = row * np.int64(N) + col
    pair = np.unique(pair)
    row = (pair // np.int64(N)).astype(np.int64)
    col = (pair %  np.int64(N)).astype(np.int64)
    order = np.argsort(row, kind="mergesort")
    row, col = row[order], col[order]
    crow = np.zeros(N+1, dtype=np.int64)
    np.add.at(crow, row+1, 1); np.cumsum(crow, out=crow)
    return torch.from_numpy(crow).cuda(), torch.from_numpy(col).cuda()

def run_once(N=8192, F=128):
    crow, col = rand_csr(N, avg_deg=16)
    x = torch.randn(N, F, device="cuda", dtype=torch.float32).contiguous()
    y_vec = torch.ops.autosage.spmm_csr(crow, col, None, x)

    # force scalar fallback
    os.environ["AUTOSAGE_VEC4"] = "0"
    y_scl = torch.ops.autosage.spmm_csr(crow, col, None, x)
    os.environ.pop("AUTOSAGE_VEC4", None)

    max_abs = (y_vec - y_scl).abs().max().item()
    print(f"F={F}: max |Δ| = {max_abs:.3e}")
    assert max_abs < 1e-4, "mismatch too large"

if __name__ == "__main__":
    for F in (64, 128, 256, 96):  # includes non-multiple-of-4
        run_once(F=F)
    print("OK")
