#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include "autosage/spmm.h"

namespace autosage {


template<int FTILE, int WARPS, bool HAS_VAL, bool VEC4>
__global__ void spmm_csr_rows_kernel(
    const int64_t* __restrict__ crow,   
    const int64_t* __restrict__ col,   
    const float*   __restrict__ val,    
    const float*   __restrict__ X,      
    float*               __restrict__ Y,
    const int32_t* __restrict__ rows,  
    int R, int F)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;        
    const int global_warp = blockIdx.x * WARPS + warp;
    if (global_warp >= R) return;

    const int row = static_cast<int>(rows[global_warp]);
    const int64_t rs = crow[row];
    const int64_t re = crow[row + 1];

    constexpr int CH = FTILE / 32;

    for (int f0 = 0; f0 < F; f0 += FTILE) {
        const int tile = min(FTILE, F - f0);

        if constexpr (VEC4) {
            const int f_rel4 = lane * 4;
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            if (f_rel4 < tile) {
                for (int64_t e = rs; e < re; ++e) {
                    const int u = static_cast<int>(col[e]);
                    const float w = HAS_VAL ? val[e] : 1.f;
                    const float4 x4 = *reinterpret_cast<const float4*>(
                        &X[static_cast<int64_t>(u) * F + (f0 + f_rel4)]
                    );
                    a0 += w * x4.x; a1 += w * x4.y; a2 += w * x4.z; a3 += w * x4.w;
                }
                const int64_t ybase = static_cast<int64_t>(row) * F + f0 + f_rel4;
                Y[ybase + 0] = a0;
                if (f_rel4 + 1 < tile) Y[ybase + 1] = a1;
                if (f_rel4 + 2 < tile) Y[ybase + 2] = a2;
                if (f_rel4 + 3 < tile) Y[ybase + 3] = a3;
            }
        } else {
            float acc[CH];
            #pragma unroll
            for (int c = 0; c < CH; ++c) acc[c] = 0.f;

            for (int64_t e = rs; e < re; ++e) {
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

            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                const int f_rel = lane + c * 32;
                const int f_abs = f0 + f_rel;
                if (f_rel < tile) {
                    Y[static_cast<int64_t>(row) * F + f_abs] = acc[c];
                }
            }
        }
    }
}


template<int FTILE, bool HAS_VAL, bool VEC4>
static void launch_rows_ft(const at::Tensor& crow, const at::Tensor& col,
                           const c10::optional<at::Tensor>& val,
                           const at::Tensor& X, at::Tensor& Y,
                           const at::Tensor& rows)
{
    constexpr int WARPS = 4;
    const int R = static_cast<int>(rows.size(0));
    if (R == 0) return;

    const int threads = WARPS * 32;
    const int blocks  = (R + WARPS - 1) / WARPS;
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t* crow_p = crow.data_ptr<int64_t>();
    const int64_t* col_p  = col.data_ptr<int64_t>();
    const float*   val_p  = HAS_VAL ? val.value().data_ptr<float>() : nullptr;
    const float*   X_p    = X.data_ptr<float>();
    float*         Y_p    = Y.data_ptr<float>();
    const int32_t* rows_p = rows.data_ptr<int32_t>();
    const int      F      = static_cast<int>(X.size(1));

    spmm_csr_rows_kernel<FTILE, WARPS, HAS_VAL, VEC4>
        <<<blocks, threads, 0, stream>>>(crow_p, col_p, val_p, X_p, Y_p, rows_p, R, F);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int FTILE>
static void launch_rows_mode(bool has_val, bool vec4,
                             const at::Tensor& crow, const at::Tensor& col,
                             const c10::optional<at::Tensor>& val,
                             const at::Tensor& X, at::Tensor& Y,
                             const at::Tensor& rows)
{
    if (vec4) {
        if (has_val) launch_rows_ft<FTILE, true,  true>(crow, col, val, X, Y, rows);
        else         launch_rows_ft<FTILE, false, true>(crow, col, val, X, Y, rows);
    } else {
        if (has_val) launch_rows_ft<FTILE, true,  false>(crow, col, val, X, Y, rows);
        else         launch_rows_ft<FTILE, false, false>(crow, col, val, X, Y, rows);
    }
}

}


namespace autosage {

void spmm_csr_rows_overwrite_launch(
    const at::Tensor& crow, const at::Tensor& col,
    const c10::optional<at::Tensor>& val,
    const at::Tensor& X, at::Tensor& Y,
    const at::Tensor& rows_i32,
    int ftile, bool use_vec4)
{
    TORCH_CHECK(crow.is_cuda() && col.is_cuda() && X.is_cuda() && Y.is_cuda() && rows_i32.is_cuda(),
                "All tensors must be CUDA for rows overwrite");
    TORCH_CHECK(crow.scalar_type()==at::kLong && col.scalar_type()==at::kLong, "CSR must be int64");
    TORCH_CHECK(rows_i32.scalar_type()==at::kInt, "rows must be int32 on CUDA");
    TORCH_CHECK(X.scalar_type()==at::kFloat && Y.scalar_type()==at::kFloat, "X/Y must be float32");
    TORCH_CHECK(X.dim()==2 && Y.dim()==2 && X.sizes()==Y.sizes(), "X and Y must be [N,F] same shape");
    if (val.has_value()) {
        TORCH_CHECK(val.value().defined() && val.value().scalar_type()==at::kFloat,
                    "val must be float32 when provided");
    }

    const bool has_val = val.has_value();
    const int FT = (ftile <= 64) ? 64 : 128;

    switch (FT) {
        case 64:  launch_rows_mode<64>(has_val, use_vec4,  crow, col, val, X, Y, rows_i32);  break;
        default:  launch_rows_mode<128>(has_val, use_vec4, crow, col, val, X, Y, rows_i32);  break;
    }
}

} 
