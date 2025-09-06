#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/util/Optional.h>

namespace autosage {
// CUDA path implemented in spmm_csr_vec4.cu
at::Tensor spmm_csr_cuda(
    const at::Tensor& crow, const at::Tensor& col,
    const c10::optional<at::Tensor>& val, const at::Tensor& X);
} // namespace autosage

namespace {
using c10::optional;

static at::Tensor spmm_csr_dispatch(const at::Tensor& crow,
                                    const at::Tensor& col,
                                    const optional<at::Tensor>& val,
                                    const at::Tensor& X) {
  TORCH_CHECK(crow.is_contiguous() && col.is_contiguous(), "CSR arrays must be contiguous");
  TORCH_CHECK(crow.scalar_type() == at::kLong && col.scalar_type() == at::kLong,
              "CSR indices must be int64 (torch.long)");
  TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
  TORCH_CHECK(X.dim() == 2, "X must be (N,F)");
  TORCH_CHECK(X.is_cuda(), "autosage::spmm_csr currently supports CUDA tensors only");
  return autosage::spmm_csr_cuda(crow, col, val, X);
}
} // anon

// Exactly one schema block per namespace.
TORCH_LIBRARY(autosage, m) {
  m.def("spmm_csr(Tensor crow, Tensor col, Tensor? val, Tensor X) -> Tensor");
}

// Device impls can be split.
TORCH_LIBRARY_IMPL(autosage, CUDA, m) {
  m.impl("spmm_csr", spmm_csr_dispatch);
}
