#include "autosage/stats.h"
#include <torch/library.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstdint>

namespace autosage {

static inline void _check_csr_inputs(const at::Tensor& crow, const at::Tensor& col) {
  TORCH_CHECK(crow.defined() && col.defined(), "crow/col must be defined");
  TORCH_CHECK(crow.dtype() == at::kLong && col.dtype() == at::kLong,
              "crow/col must be int64 (Long)");
  TORCH_CHECK(crow.dim() == 1 && col.dim() == 1, "crow/col must be 1-D");
  TORCH_CHECK(crow.size(0) >= 2, "crow must have length >= 2");
  TORCH_CHECK(crow.device().type() == col.device().type(),
              "crow and col must be on the same device");
  if (crow.device().is_cpu()) {
    const int64_t* crow_a = crow.data_ptr<int64_t>();
    const int64_t nnz = crow_a[crow.size(0) - 1];
    TORCH_CHECK(nnz == col.size(0),
      "CSR crow last value (", nnz, ") != col.size(0) (", col.size(0), ")");
  }
}

template <typename T>
static inline double quantile_inplace(std::vector<T>& v, double q) {
  if (v.empty()) return 0.0;
  q = std::min(std::max(q, 0.0), 1.0);
  size_t n = v.size();
  size_t idx = static_cast<size_t>(std::floor(q * (n - 1)));
  std::nth_element(v.begin(), v.begin() + idx, v.end());
  return static_cast<double>(v[idx]);
}

at::Tensor compute_graph_stats(const at::Tensor& crow, const at::Tensor& col) {
  _check_csr_inputs(crow, col);

  at::Tensor crow_cpu = crow.is_cpu() ? crow : crow.to(at::kCPU, /*non_blocking=*/true);

  const int64_t* cr = crow_cpu.data_ptr<int64_t>();
  const int64_t N   = crow_cpu.size(0) - 1;
  const int64_t nnz = cr[N];

  std::vector<int64_t> degs(static_cast<size_t>(N));
  int64_t max_deg = 0;
  for (int64_t i = 0; i < N; ++i) {
    int64_t d = cr[i + 1] - cr[i];
    degs[static_cast<size_t>(i)] = d;
    if (d > max_deg) max_deg = d;
  }

  const double mean_deg = (N > 0) ? static_cast<double>(nnz) / static_cast<double>(N) : 0.0;

  auto qv = degs; const double q50 = quantile_inplace(qv, 0.50);
  qv = degs;      const double q90 = quantile_inplace(qv, 0.90);
  qv = degs;      const double q99 = quantile_inplace(qv, 0.99);

  const double heavy_tail = q99 / std::max(1.0, q50);

  at::Tensor out = at::empty({8}, at::device(at::kCPU).dtype(at::kFloat));
  auto* o = out.data_ptr<float>();
  o[0] = static_cast<float>(N);
  o[1] = static_cast<float>(nnz);
  o[2] = static_cast<float>(mean_deg);
  o[3] = static_cast<float>(max_deg);
  o[4] = static_cast<float>(q50);
  o[5] = static_cast<float>(q90);
  o[6] = static_cast<float>(q99);
  o[7] = static_cast<float>(heavy_tail);
  return out;
}

at::Tensor get_device_info(int64_t device_index) {
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  TORCH_CHECK(e == cudaSuccess, "cudaGetDeviceCount failed: ", cudaGetErrorString(e));
  TORCH_CHECK(device_index >= 0 && device_index < count,
              "device_index out of range [0, ", count, ")");

  int dev = static_cast<int>(device_index);
  cudaDeviceProp prop{};
  e = cudaGetDeviceProperties(&prop, dev);
  TORCH_CHECK(e == cudaSuccess, "cudaGetDeviceProperties failed: ", cudaGetErrorString(e));

  at::Tensor out = at::empty({7}, at::device(at::kCPU).dtype(at::kLong));
  auto* o = out.data_ptr<int64_t>();
  o[0] = static_cast<int64_t>(prop.multiProcessorCount);
  o[1] = static_cast<int64_t>(prop.major);
  o[2] = static_cast<int64_t>(prop.minor);
  o[3] = static_cast<int64_t>(prop.l2CacheSize);
  o[4] = static_cast<int64_t>(prop.sharedMemPerMultiprocessor);
  o[5] = static_cast<int64_t>(prop.regsPerMultiprocessor);
  o[6] = static_cast<int64_t>(prop.warpSize);
  return out;
}

// ---- Operator schemas (fragment to avoid duplicate TORCH_LIBRARY)
TORCH_LIBRARY_FRAGMENT(autosage, m) {
  m.def("compute_graph_stats(Tensor crow, Tensor col) -> Tensor");
  m.def("get_device_info(int device_index) -> Tensor");
}

// ---- Implementations
static at::Tensor compute_graph_stats_dispatch(const at::Tensor& crow, const at::Tensor& col) {
  return compute_graph_stats(crow, col);
}
static at::Tensor get_device_info_dispatch(int64_t device_index) {
  return get_device_info(device_index);
}

// Tensor-dependent op: register for CPU/CUDA
TORCH_LIBRARY_IMPL(autosage, CPU, m) {
  m.impl("compute_graph_stats", TORCH_FN(compute_graph_stats_dispatch));
}
TORCH_LIBRARY_IMPL(autosage, CUDA, m) {
  m.impl("compute_graph_stats", TORCH_FN(compute_graph_stats_dispatch));
}

// No-tensor-input op: register under CatchAll
TORCH_LIBRARY_IMPL(autosage, CatchAll, m) {
  m.impl("get_device_info", TORCH_FN(get_device_info_dispatch));
}

} // namespace autosage
