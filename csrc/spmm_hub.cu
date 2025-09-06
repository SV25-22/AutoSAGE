#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "autosage/spmm.h"

namespace autosage {

// ---------------------------- Kernel ----------------------------
// CTA-per-hub: one block per heavy row, WARPS warps split neighbor list,
// accumulate into shared memory [WARPS x FTILE], warp0 reduces across WARPS,
// and overwrites Y[row, :] once per tile.

template<int FTILE, int WARPS, bool HAS_VAL, bool VEC4>
__global__ void spmm_csr_hub_overwrite_kernel(
    const int64_t* __restrict__ crow,   // [N+1]
    const int64_t* __restrict__ col,    // [nnz]
    const float*   __restrict__ val,    // [nnz] or nullptr
    const float*   __restrict__ X,      // [N, F]
    float*               __restrict__ Y,// [N, F]
    const int32_t* __restrict__ heavy_rows, // [H]
    int F)
{
    extern __shared__ float smem[]; // WARPS * FTILE
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;            // [0, WARPS)
    const int row   = static_cast<int>(heavy_rows[blockIdx.x]);

    const int64_t row_start = crow[row];
    const int64_t row_end   = crow[row + 1];

    // Scalar path: lanes cover f_rel = lane + c*32
    constexpr int CH = FTILE / 32;

    for (int f0 = 0; f0 < F; f0 += FTILE) {
        const int tile = min(FTILE, F - f0);

        if constexpr (VEC4) {
            // Vectorized per-lane accumulation: each lane owns 4 contiguous features
            // starting at f_rel4 = lane*4. For FTILE<=128, f_rel4 advances at most once.
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            const int f_rel4 = lane * 4;
            if (f_rel4 < tile) {
                for (int64_t e = row_start + warp; e < row_end; e += WARPS) {
                    const int u = static_cast<int>(col[e]);
                    const float w = HAS_VAL ? val[e] : 1.f;
                    const float4 x4 = *reinterpret_cast<const float4*>(
                        &X[static_cast<int64_t>(u) * F + (f0 + f_rel4)]
                    );
                    a0 += w * x4.x; a1 += w * x4.y; a2 += w * x4.z; a3 += w * x4.w;
                }
                // write this warp's partials into smem
                float* sm = smem + warp * FTILE;
                sm[f_rel4 + 0] = a0;
                if (f_rel4 + 1 < tile) sm[f_rel4 + 1] = a1;
                if (f_rel4 + 2 < tile) sm[f_rel4 + 2] = a2;
                if (f_rel4 + 3 < tile) sm[f_rel4 + 3] = a3;
            }
        } else {
            // Scalar path (stable & general)
            float acc[CH];
            #pragma unroll
            for (int c = 0; c < CH; ++c) acc[c] = 0.f;

            for (int64_t e = row_start + warp; e < row_end; e += WARPS) {
                const int u = static_cast<int>(col[e]);
                const float w = HAS_VAL ? val[e] : 1.f;
                #pragma unroll
                for (int c = 0; c < CH; ++c) {
                    const int f_rel = lane + c * 32;
                    const int f_abs = f0 + f_rel;
                    if (f_rel < tile) {
                        acc[c] += w * __ldg(&X[static_cast<int64_t>(u) * F + f_abs]);
                    }
                }
            }
            float* sm = smem + warp * FTILE;
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                const int f_rel = lane + c * 32;
                if (f_rel < tile) sm[f_rel] = acc[c];
            }
        }

        __syncthreads();

        // Inter-warp reduction + single write
        if (warp == 0) {
            for (int f_rel = lane; f_rel < tile; f_rel += 32) {
                float sum = 0.f;
                #pragma unroll
                for (int w = 0; w < WARPS; ++w) {
                    sum += smem[w * FTILE + f_rel];
                }
                Y[static_cast<int64_t>(row) * F + (f0 + f_rel)] = sum;
            }
        }
        __syncthreads();
    }
}

// ------------------------- Launch helpers -------------------------

template<int FTILE, int WARPS, bool VEC4>
static void dispatch_hasval(bool has_val,
    const at::Tensor& crow, const at::Tensor& col,
    const c10::optional<at::Tensor>& val_opt,
    const at::Tensor& X, at::Tensor& Y,
    const at::Tensor& heavy_rows)
{
    const int H = static_cast<int>(heavy_rows.size(0));
    if (H == 0) return;

    const int threads = WARPS * 32;
    const dim3 grid(H);
    const dim3 block(threads);
    const size_t smem_bytes = static_cast<size_t>(WARPS) * FTILE * sizeof(float);

    const int64_t* crow_p = crow.data_ptr<int64_t>();
    const int64_t* col_p  = col.data_ptr<int64_t>();
    const float* val_p    = (has_val ? val_opt.value().data_ptr<float>() : nullptr);
    const float* X_p      = X.data_ptr<float>();
    float* Y_p            = Y.data_ptr<float>();
    const int32_t* heavy_p= heavy_rows.data_ptr<int32_t>();
    const int F           = static_cast<int>(X.size(1));
    auto stream = at::cuda::getCurrentCUDAStream();

    if (has_val) {
        spmm_csr_hub_overwrite_kernel<FTILE, WARPS, true, VEC4>
            <<<grid, block, smem_bytes, stream>>>(crow_p, col_p, val_p, X_p, Y_p, heavy_p, F);
    } else {
        spmm_csr_hub_overwrite_kernel<FTILE, WARPS, false, VEC4>
            <<<grid, block, smem_bytes, stream>>>(crow_p, col_p, nullptr, X_p, Y_p, heavy_p, F);
    }
}

template<int FTILE, bool VEC4>
static void dispatch_warps(int wpb, bool has_val,
    const at::Tensor& crow, const at::Tensor& col,
    const c10::optional<at::Tensor>& val_opt,
    const at::Tensor& X, at::Tensor& Y,
    const at::Tensor& heavy_rows)
{
    int W = (wpb <= 2) ? 2 : (wpb <= 4 ? 4 : 8);
    switch (W) {
        case 2: dispatch_hasval<FTILE, 2, VEC4>(has_val, crow, col, val_opt, X, Y, heavy_rows); break;
        case 4: dispatch_hasval<FTILE, 4, VEC4>(has_val, crow, col, val_opt, X, Y, heavy_rows); break;
        default: dispatch_hasval<FTILE, 8, VEC4>(has_val, crow, col, val_opt, X, Y, heavy_rows); break;
    }
}

} // namespace autosage

// --------------------- Public launcher (called from split) ---------------------

namespace autosage {

void spmm_csr_hub_overwrite_launch(
    const at::Tensor& crow, const at::Tensor& col,
    const c10::optional<at::Tensor>& val,
    const at::Tensor& X, at::Tensor& Y,
    const at::Tensor& heavy_rows,
    int ftile, int wpb, bool use_vec4)
{
    TORCH_CHECK(crow.is_cuda() && col.is_cuda() && X.is_cuda() && Y.is_cuda() && heavy_rows.is_cuda(),
                "All tensors must be CUDA for hub overwrite");
    TORCH_CHECK(crow.scalar_type()==at::kLong && col.scalar_type()==at::kLong, "CSR must be int64");
    TORCH_CHECK(heavy_rows.scalar_type()==at::kInt, "heavy_rows must be int32 on CUDA");
    TORCH_CHECK(X.scalar_type()==at::kFloat && Y.scalar_type()==at::kFloat, "X/Y must be float32");
    TORCH_CHECK(X.dim()==2 && Y.dim()==2 && X.size(0)==Y.size(0) && X.size(1)==Y.size(1),
                "X and Y must be [N,F] with same shape");
    if (val.has_value()) {
        TORCH_CHECK(val.value().defined() && val.value().scalar_type()==at::kFloat,
                    "val must be float32 when provided");
    }

    const bool has_val = val.has_value();
    const int FT = (ftile <= 64) ? 64 : 128;

    if (use_vec4) {
        switch (FT) {
            case 64:  dispatch_warps<64,  true>(wpb, has_val, crow, col, val, X, Y, heavy_rows);  break;
            default:  dispatch_warps<128, true>(wpb, has_val, crow, col, val, X, Y, heavy_rows); break;
        }
    } else {
        switch (FT) {
            case 64:  dispatch_warps<64,  false>(wpb, has_val, crow, col, val, X, Y, heavy_rows);  break;
            default:  dispatch_warps<128, false>(wpb, has_val, crow, col, val, X, Y, heavy_rows); break;
        }
    }
}

} // namespace autosage
