#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "autosage/spmm.h"

namespace {

// Simple ceildiv
__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Each warp processes one row; feature dimension is processed in tiles of FTILE.
template<int FTILE>
__global__ void csr_spmm_warp_per_row_kernel(
    const long* __restrict__ crow,       // [N+1]
    const long* __restrict__ col,        // [nnz]
    const float* __restrict__ val,       // [nnz] or nullptr
    const float* __restrict__ X,         // [N, F]
    float* __restrict__ Y,               // [N, F]
    int N, int F)
{
    const int warp_id_in_block = threadIdx.x >> 5; // /32
    const int lane = threadIdx.x & 31;
    const int WARPS_PER_BLOCK = blockDim.x >> 5;

    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    if (row >= N) return;

    const long start = crow[row];
    const long end   = crow[row + 1];

    // Feature tiles
    for (int f0 = 0; f0 < F; f0 += FTILE) {
        const int tile = (f0 + FTILE <= F) ? FTILE : (F - f0);

        // Each lane owns features f = f0 + lane, f += 32
        for (int f = lane; f < tile; f += 32) {
            float acc = 0.0f;
            const int feat = f0 + f;
            // Sum over neighbors
            for (long e = start; e < end; ++e) {
                const int c = static_cast<int>(col[e]);
                const float w = (val ? val[e] : 1.0f);
                acc += w * X[c * F + feat];
            }
            Y[row * F + feat] = acc;
        }
    }
}

template<int FTILE>
void launch_kernel(const at::Tensor& crow,
                   const at::Tensor& col,
                   const c10::optional<at::Tensor>& val,
                   const at::Tensor& X,
                   at::Tensor& Y)
{
    const int N = static_cast<int>(X.size(0));
    const int F = static_cast<int>(X.size(1));
    const int WARPS_PER_BLOCK = 4;
    const dim3 block(WARPS_PER_BLOCK * 32);
    const dim3 grid(ceil_div(N, WARPS_PER_BLOCK));

    const long* crow_p = crow.data_ptr<long>();
    const long* col_p  = col.data_ptr<long>();
    const float* val_p = (val && val->defined()) ? val->data_ptr<float>() : nullptr;
    const float* X_p   = X.data_ptr<float>();
    float* Y_p         = Y.data_ptr<float>();

    csr_spmm_warp_per_row_kernel<FTILE><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        crow_p, col_p, val_p, X_p, Y_p, N, F
    );
}

} // anonymous namespace

namespace autosage {

at::Tensor spmm_csr_cuda(const at::Tensor& crow,
                         const at::Tensor& col,
                         const c10::optional<at::Tensor>& val,
                         const at::Tensor& X)
{
    TORCH_CHECK(X.is_cuda(), "spmm_csr_cuda: X must be a CUDA tensor.");
    TORCH_CHECK(crow.is_cuda() && col.is_cuda(),
                "spmm_csr_cuda: crow and col must be CUDA tensors.");
    TORCH_CHECK((!val.has_value() || !val.value().defined()) || val.value().is_cuda(),
                "spmm_csr_cuda: val must be CUDA if provided.");

    TORCH_CHECK(crow.scalar_type() == at::kLong && col.scalar_type() == at::kLong,
                "CSR indices must be int64 (torch.long).");
    TORCH_CHECK(X.scalar_type() == at::kFloat, "P1 supports float32 features only.");
    TORCH_CHECK(crow.dim() == 1 && col.dim() == 1, "CSR crow/col must be 1D.");
    TORCH_CHECK(X.dim() == 2, "X must be [N, F].");

    // Make contiguous copies if needed
    auto Xc    = X.contiguous();
    auto crowc = crow.contiguous();
    auto colc  = col.contiguous();
    c10::optional<at::Tensor> valc = c10::nullopt;
    if (val.has_value() && val.value().defined()) {
        TORCH_CHECK(val.value().scalar_type() == at::kFloat, "P1 supports float32 edge weights only.");
        valc = val.value().contiguous();
    }

    const int64_t N   = Xc.size(0);
    const int64_t F   = Xc.size(1);
    const int64_t nnz = colc.size(0);

    TORCH_CHECK(crowc.size(0) == N + 1, "crow length must be N+1 (got %ld vs %ld).",
                (long)crowc.size(0), (long)(N + 1));
    const long last_ptr = crowc.index({N}).item<long>();
    TORCH_CHECK(last_ptr == nnz, "crow[N] must equal nnz=col.size(0) (got %ld vs %ld).",
                last_ptr, (long)nnz);
    if (valc && valc->defined()) {
        TORCH_CHECK(valc->size(0) == nnz, "val length must equal nnz (got %ld vs %ld).",
                    (long)valc->size(0), (long)nnz);
    }

    auto Y = at::zeros_like(Xc);

    // Pick a simple feature tile
    if      (F >= 128) launch_kernel<128>(crowc, colc, valc, Xc, Y);
    else if (F >= 64)  launch_kernel<64 >(crowc, colc, valc, Xc, Y);
    else               launch_kernel<32 >(crowc, colc, valc, Xc, Y);

    return Y;
}

} // namespace autosage
