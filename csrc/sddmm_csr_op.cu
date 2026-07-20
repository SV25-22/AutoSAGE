#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

__inline__ __device__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void sddmm_rows_kernel(
    const int64_t* __restrict__ crow,
    const int64_t* __restrict__ col,
    const float* __restrict__ query,
    const float* __restrict__ key,
    float* __restrict__ output,
    int rows,
    int features,
    int hub_threshold) {
  const int lane = threadIdx.x & 31;
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= rows) {
    return;
  }
  const int64_t begin = crow[row];
  const int64_t end = crow[row + 1];
  if (end - begin >= hub_threshold) {
    return;
  }
  const float* query_row = query + static_cast<int64_t>(row) * features;
  for (int64_t edge = begin; edge < end; ++edge) {
    const float* key_row = key + col[edge] * static_cast<int64_t>(features);
    float sum = 0.0f;
    for (int feature = lane; feature < features; feature += 32) {
      sum += query_row[feature] * key_row[feature];
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
      output[edge] = sum;
    }
  }
}

__global__ void sddmm_hubs_kernel(
    const int64_t* __restrict__ crow,
    const int64_t* __restrict__ col,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const int32_t* __restrict__ hub_rows,
    float* __restrict__ output,
    int features,
    int hub_count) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int hub_index = blockIdx.x;
  if (hub_index >= hub_count) {
    return;
  }
  const int row = hub_rows[hub_index];
  const int64_t begin = crow[row];
  const int64_t end = crow[row + 1];
  const float* query_row = query + static_cast<int64_t>(row) * features;
  for (int64_t edge = begin + warp; edge < end; edge += warps) {
    const float* key_row = key + col[edge] * static_cast<int64_t>(features);
    float sum = 0.0f;
    for (int feature = lane; feature < features; feature += 32) {
      sum += query_row[feature] * key_row[feature];
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
      output[edge] = sum;
    }
  }
}

void check_sddmm_inputs(
    const at::Tensor& crow,
    const at::Tensor& col,
    const at::Tensor& query,
    const at::Tensor& key) {
  TORCH_CHECK(crow.is_cuda() && col.is_cuda() && query.is_cuda() && key.is_cuda(),
              "all SDDMM inputs must be CUDA tensors");
  TORCH_CHECK(crow.scalar_type() == at::kLong && col.scalar_type() == at::kLong,
              "crow and col must use torch.long indices");
  TORCH_CHECK(query.scalar_type() == at::kFloat && key.scalar_type() == at::kFloat,
              "query and key must be float32");
  TORCH_CHECK(query.dim() == 2 && query.sizes() == key.sizes(),
              "query and key must have the same [N, F] shape");
  TORCH_CHECK(crow.numel() == query.size(0) + 1, "crow must contain N + 1 entries");
  TORCH_CHECK(crow.device() == query.device() && col.device() == query.device() && key.device() == query.device(),
              "all SDDMM inputs must be on the same CUDA device");
  TORCH_CHECK(crow.is_contiguous() && col.is_contiguous() && query.is_contiguous() && key.is_contiguous(),
              "all SDDMM inputs must be contiguous");
  TORCH_CHECK(query.size(0) <= std::numeric_limits<int>::max() &&
                  query.size(1) <= std::numeric_limits<int>::max(),
              "SDDMM dimensions exceed the kernel index range");
}

at::Tensor sddmm_csr(
    const at::Tensor& crow,
    const at::Tensor& col,
    const at::Tensor& query,
    const at::Tensor& key) {
  check_sddmm_inputs(crow, col, query, key);
  const int rows = static_cast<int>(query.size(0));
  const int features = static_cast<int>(query.size(1));
  auto output = at::empty({col.numel()}, query.options());
  if (rows == 0) {
    return output;
  }
  constexpr int warps = 4;
  const dim3 block(warps * 32);
  const dim3 grid((rows + warps - 1) / warps);
  sddmm_rows_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      crow.data_ptr<int64_t>(),
      col.data_ptr<int64_t>(),
      query.data_ptr<float>(),
      key.data_ptr<float>(),
      output.data_ptr<float>(),
      rows,
      features,
      std::numeric_limits<int>::max());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sddmm_csr_split(
    const at::Tensor& crow,
    const at::Tensor& col,
    const at::Tensor& query,
    const at::Tensor& key,
    int64_t warps_per_block,
    int64_t hub_threshold) {
  check_sddmm_inputs(crow, col, query, key);
  TORCH_CHECK(warps_per_block == 2 || warps_per_block == 4 || warps_per_block == 8,
              "warps_per_block must be 2, 4, or 8");
  TORCH_CHECK(hub_threshold > 0 && hub_threshold <= std::numeric_limits<int>::max(),
              "hub_threshold must be a positive 32-bit integer");
  const int rows = static_cast<int>(query.size(0));
  const int features = static_cast<int>(query.size(1));
  auto output = at::empty({col.numel()}, query.options());
  if (rows == 0) {
    return output;
  }

  constexpr int row_warps = 4;
  const dim3 row_block(row_warps * 32);
  const dim3 row_grid((rows + row_warps - 1) / row_warps);
  sddmm_rows_kernel<<<row_grid, row_block, 0, at::cuda::getCurrentCUDAStream()>>>(
      crow.data_ptr<int64_t>(),
      col.data_ptr<int64_t>(),
      query.data_ptr<float>(),
      key.data_ptr<float>(),
      output.data_ptr<float>(),
      rows,
      features,
      static_cast<int>(hub_threshold));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const auto pointer_count = crow.size(0);
  auto degrees = crow.slice(0, 1, pointer_count) - crow.slice(0, 0, pointer_count - 1);
  auto hub_rows = at::nonzero(degrees.ge(hub_threshold)).flatten().to(at::kInt).contiguous();
  const int hub_count = static_cast<int>(hub_rows.numel());
  if (hub_count > 0) {
    const int warps = static_cast<int>(warps_per_block);
    sddmm_hubs_kernel<<<hub_count, warps * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        crow.data_ptr<int64_t>(),
        col.data_ptr<int64_t>(),
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        hub_rows.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        features,
        hub_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

}

TORCH_LIBRARY_FRAGMENT(autosage, library) {
  library.def("sddmm_csr(Tensor crow, Tensor col, Tensor query, Tensor key) -> Tensor");
  library.def(
      "sddmm_csr_split(Tensor crow, Tensor col, Tensor query, Tensor key, "
      "int warps_per_block, int hub_threshold) -> Tensor");
}

TORCH_LIBRARY_IMPL(autosage, CUDA, library) {
  library.impl("sddmm_csr", sddmm_csr);
  library.impl("sddmm_csr_split", sddmm_csr_split);
}
