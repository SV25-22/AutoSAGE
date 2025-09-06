// csrc/spmm_csr_vec4.cu
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

using at::Tensor;

#define CUDA_CHECK() do { \
  cudaError_t err = cudaGetLastError(); \
  TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
} while (0)

static __forceinline__ __device__ int warp_id_in_block() { return threadIdx.x / 32; }
static __forceinline__ __device__ int lane_id()          { return threadIdx.x & 31; }

// ------------------------------------
// Scalar (fallback) warp-per-row kernel
// ------------------------------------
template <bool HasVal>
__global__ void spmm_csr_scalar_kernel(
    const int64_t* __restrict__ crow,  // (N+1)
    const int64_t* __restrict__ col,   // (E)
    const float*   __restrict__ val,   // (E) or nullptr
    const float*   __restrict__ X,     // (N,F) row-major
    float*         __restrict__ Y,     // (N,F)
    int N, int F)
{
    const int warps_per_block = blockDim.x / 32;
    const int warp_global = blockIdx.x * warps_per_block + warp_id_in_block();
    if (warp_global >= N) return;

    const int row  = warp_global;
    const int lane = lane_id();

    const int start = static_cast<int>(crow[row]);
    const int end   = static_cast<int>(crow[row + 1]);

    for (int f = lane; f < F; f += 32) {
        float acc = 0.f;
        for (int p = start; p < end; ++p) {
            const int s = static_cast<int>(col[p]);
            const float w = HasVal ? val[p] : 1.f;
            acc += w * X[s * F + f];
        }
        Y[row * F + f] = acc;
    }
}

// -----------------------------
// Vec4 warp-per-row (F % 4 == 0)
// -----------------------------
template <bool HasVal>
__global__ void spmm_csr_vec4_kernel(
    const int64_t* __restrict__ crow,
    const int64_t* __restrict__ col,
    const float*   __restrict__ val,
    const float*   __restrict__ X,
    float*         __restrict__ Y,
    int N, int F)
{
    const int F4 = F >> 2; // float4 groups per row
    const int warps_per_block = blockDim.x / 32;
    const int warp_global = blockIdx.x * warps_per_block + warp_id_in_block();
    if (warp_global >= N) return;

    const int row  = warp_global;
    const int lane = lane_id();

    const int start = static_cast<int>(crow[row]);
    const int end   = static_cast<int>(crow[row + 1]);

    float4* __restrict__ y4 = reinterpret_cast<float4*>(Y + row * F);

    for (int f4 = lane; f4 < F4; f4 += 32) {
        float4 acc = {0.f, 0.f, 0.f, 0.f};
        for (int p = start; p < end; ++p) {
            const int s = static_cast<int>(col[p]);
            const float w = HasVal ? val[p] : 1.f;
            const float4* __restrict__ x4 = reinterpret_cast<const float4*>(X + s * F);
            const float4 xv = x4[f4];
            acc.x += w * xv.x;
            acc.y += w * xv.y;
            acc.z += w * xv.z;
            acc.w += w * xv.w;
        }
        y4[f4] = acc;
    }
}

// -----------------------------
// Host dispatch (CUDA path)
// -----------------------------
static inline bool is_aligned_16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0u;
}

namespace autosage {

at::Tensor spmm_csr_cuda(
    const Tensor& crow, const Tensor& col,
    const c10::optional<Tensor>& val_opt, const Tensor& x)
{
    TORCH_CHECK(crow.is_cuda() && col.is_cuda() && x.is_cuda(), "spmm_csr: tensors must be CUDA");
    TORCH_CHECK(crow.scalar_type() == at::kLong && col.scalar_type() == at::kLong,
                "crow/col must be int64 (torch.long)");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(crow.is_contiguous() && col.is_contiguous() && x.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be (N,F)");

    const int64_t N64 = x.size(0);
    const int64_t F64 = x.size(1);
    TORCH_CHECK(crow.size(0) == N64 + 1, "crow must have N+1 elements");

    const int N = static_cast<int>(N64);
    const int F = static_cast<int>(F64);

    const float* x_ptr = x.data_ptr<float>();
    Tensor y = at::empty_like(x);
    float* y_ptr = y.data_ptr<float>();

    const float* val_ptr = nullptr;
    if (val_opt.has_value() && val_opt->defined()) {
        TORCH_CHECK(val_opt->scalar_type() == at::kFloat && val_opt->is_cuda(), "val must be CUDA float32");
        TORCH_CHECK(val_opt->is_contiguous(), "val must be contiguous");
        val_ptr = val_opt->data_ptr<float>();
    }

    // Env toggle (default ON)
    bool want_vec4 = true;
    if (const char* env = std::getenv("AUTOSAGE_VEC4")) {
        const std::string v(env);
        want_vec4 = !(v == "0" || v == "false" || v == "False");
    }

    const bool vec4_ok = want_vec4 &&
                         (F % 4 == 0) &&
                         is_aligned_16(x_ptr) &&
                         is_aligned_16(y_ptr);

    const int warps_per_block = 4;
    const dim3 block(warps_per_block * 32);
    const dim3 grid((N + warps_per_block - 1) / warps_per_block);

    const int64_t* crow_p = crow.data_ptr<int64_t>();
    const int64_t* col_p  = col.data_ptr<int64_t>();

    if (vec4_ok) {
        if (val_ptr) {
            spmm_csr_vec4_kernel<true><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                crow_p, col_p, val_ptr, x_ptr, y_ptr, N, F);
        } else {
            spmm_csr_vec4_kernel<false><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                crow_p, col_p, nullptr, x_ptr, y_ptr, N, F);
        }
    } else {
        if (val_ptr) {
            spmm_csr_scalar_kernel<true><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                crow_p, col_p, val_ptr, x_ptr, y_ptr, N, F);
        } else {
            spmm_csr_scalar_kernel<false><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                crow_p, col_p, nullptr, x_ptr, y_ptr, N, F);
        }
    }
    CUDA_CHECK();
    return y;
}

} // namespace autosage

// No TORCH_LIBRARY blocks here; registration is in spmm_binding.cpp.
