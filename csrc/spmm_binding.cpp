#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "autosage/spmm.h"

namespace {

void check_inputs(
    const at::Tensor& crow,
    const at::Tensor& col,
    const c10::optional<at::Tensor>& val,
    const at::Tensor& x) {
  TORCH_CHECK(crow.is_cuda() && col.is_cuda() && x.is_cuda(), "crow, col, and x must be CUDA tensors");
  TORCH_CHECK(crow.scalar_type() == at::kLong && col.scalar_type() == at::kLong,
              "crow and col must use torch.long indices");
  TORCH_CHECK(crow.dim() == 1 && col.dim() == 1, "crow and col must be one-dimensional");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "x must be contiguous with shape [N, F]");
  TORCH_CHECK(crow.is_contiguous() && col.is_contiguous(), "crow and col must be contiguous");
  TORCH_CHECK(crow.device() == x.device() && col.device() == x.device(),
              "crow, col, and x must be on the same CUDA device");
  TORCH_CHECK(crow.numel() == x.size(0) + 1, "crow must contain N + 1 entries");
  if (val.has_value() && val->defined()) {
    TORCH_CHECK(val->is_cuda() && val->device() == x.device(), "val must be on the same CUDA device as x");
    TORCH_CHECK(val->scalar_type() == at::kFloat && val->is_contiguous(),
                "val must be contiguous float32");
    TORCH_CHECK(val->numel() == col.numel(), "val must contain one weight per nonzero");
  }
}

at::Tensor spmm_csr_dispatch(
    const at::Tensor& crow,
    const at::Tensor& col,
    const c10::optional<at::Tensor>& val,
    const at::Tensor& x) {
  check_inputs(crow, col, val, x);
  return autosage::spmm_csr_cuda(crow, col, val, x);
}

at::Tensor spmm_csr_split_dispatch(
    const at::Tensor& crow,
    const at::Tensor& col,
    const c10::optional<at::Tensor>& val,
    const at::Tensor& x,
    int64_t feature_tile,
    int64_t warps_per_block,
    int64_t hub_threshold) {
  check_inputs(crow, col, val, x);
  TORCH_CHECK(feature_tile == 64 || feature_tile == 128, "feature_tile must be 64 or 128");
  TORCH_CHECK(warps_per_block == 2 || warps_per_block == 4 || warps_per_block == 8,
              "warps_per_block must be 2, 4, or 8");
  TORCH_CHECK(hub_threshold > 0, "hub_threshold must be positive");
  return autosage::spmm_csr_split_cuda(
      crow,
      col,
      val,
      x,
      static_cast<int>(feature_tile),
      static_cast<int>(warps_per_block),
      static_cast<int>(hub_threshold));
}

}

TORCH_LIBRARY(autosage, library) {
  library.def("spmm_csr(Tensor crow, Tensor col, Tensor? val, Tensor x) -> Tensor");
  library.def(
      "spmm_csr_split(Tensor crow, Tensor col, Tensor? val, Tensor x, "
      "int feature_tile, int warps_per_block, int hub_threshold) -> Tensor");
}

TORCH_LIBRARY_IMPL(autosage, CUDA, library) {
  library.impl("spmm_csr", spmm_csr_dispatch);
  library.impl("spmm_csr_split", spmm_csr_split_dispatch);
}
