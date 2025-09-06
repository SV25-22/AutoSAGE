#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <c10/cuda/CUDAStream.h>  // <-- current stream API

namespace {

// y[r, :] = sum_{j in row r} x[col[j], :]
template <bool WithVal>
__global__ void spmm_vec4_kernel(
    const int64_t* __restrict__ crow,  // [N+1]
    const int64_t* __restrict__ col,   // [nnz]
    const float*   __restrict__ val,   // [nnz] (nullable if !WithVal)
    const float*   __restrict__ x,     // [N, F]
    float*         __restrict__ y,     // [N, F]
    int64_t N, int64_t F, int64_t nF4)
{
    const int r = blockIdx.x;                 // warp-per-row
    const int lane = threadIdx.x & 31;
    if (r >= N) return;

    const int64_t row_start = crow[r];
    const int64_t row_end   = crow[r+1];

    for (int64_t f4 = lane; f4 < nF4; f4 += 32) {
        float4 acc = {0.f, 0.f, 0.f, 0.f};
        for (int64_t jj = row_start; jj < row_end; ++jj) {
            const int c = static_cast<int>(col[jj]);
            const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x + (int64_t)c * F);
            float4 v = x4[f4];
            if constexpr (WithVal) {
                const float w = val[jj];
                acc.x += w * v.x; acc.y += w * v.y; acc.z += w * v.z; acc.w += w * v.w;
            } else {
                acc.x += v.x;    acc.y += v.y;    acc.z += v.z;    acc.w += v.w;
            }
        }
        float* __restrict__ yrow = y + (int64_t)r * F + 4 * f4;
        yrow[0] = acc.x; yrow[1] = acc.y; yrow[2] = acc.z; yrow[3] = acc.w;
    }
}

} // namespace

extern "C" void autosage_spmm_vec4_launch(
    const at::Tensor& crow, const at::Tensor& col, const at::Tensor& val_opt,
    const at::Tensor& x, at::Tensor& y)
{
    const int64_t N = x.size(0);
    const int64_t F = x.size(1);
    const int64_t nF4 = F / 4;

    const dim3 block(32);
    const dim3 grid(static_cast<unsigned>(N));

    const int64_t* crow_p = crow.data_ptr<int64_t>();
    const int64_t* col_p  = col.data_ptr<int64_t>();
    const float*   x_p    = x.data_ptr<float>();
    float*         y_p    = y.data_ptr<float>();

    auto stream = c10::cuda::getCurrentCUDAStream();

    if (val_opt.defined()) {
        const float* v_p = val_opt.data_ptr<float>();
        spmm_vec4_kernel<true><<<grid, block, 0, stream.stream()>>>(
            crow_p, col_p, v_p, x_p, y_p, N, F, nF4);
    } else {
        spmm_vec4_kernel<false><<<grid, block, 0, stream.stream()>>>(
            crow_p, col_p, nullptr, x_p, y_p, N, F, nF4);
    }
    // errors propagate at next sync
}
