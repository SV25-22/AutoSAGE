#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/util/Optional.h>
#include "autosage/spmm.h"

namespace autosage {
// keep declaration available (implemented in spmm_csr_vec4.cu)
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
  TORCH_CHECK(crow.scalar_type()==at::kLong && col.scalar_type()==at::kLong,
              "CSR indices must be int64 (torch.long)");
  TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
  TORCH_CHECK(X.dim()==2, "X must be (N,F)");
  TORCH_CHECK(X.is_cuda(), "autosage::spmm_csr currently supports CUDA tensors only");
  return autosage::spmm_csr_cuda(crow, col, val, X);
}

static at::Tensor spmm_csr_split_dispatch(const at::Tensor& crow,
                                          const at::Tensor& col,
                                          const optional<at::Tensor>& val,
                                          const at::Tensor& X,
                                          int64_t ftile, int64_t wpb, int64_t hubT) {
  TORCH_CHECK(crow.is_contiguous() && col.is_contiguous(), "CSR arrays must be contiguous");
  TORCH_CHECK(crow.scalar_type()==at::kLong && col.scalar_type()==at::kLong,
              "CSR indices must be int64 (torch.long)");
  TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
  TORCH_CHECK(X.dim()==2, "X must be (N,F)");
  TORCH_CHECK(X.is_cuda(), "autosage::spmm_csr_split supports CUDA tensors only");
  return autosage::spmm_csr_split_cuda(crow, col, val, X,
                                       static_cast<int>(ftile),
                                       static_cast<int>(wpb),
                                       static_cast<int>(hubT));
}

} // anon

TORCH_LIBRARY(autosage, m) {
  m.def("spmm_csr(Tensor crow, Tensor col, Tensor? val, Tensor X) -> Tensor");
  m.def("spmm_csr_split(Tensor crow, Tensor col, Tensor? val, Tensor X, int ftile, int wpb, int hubT) -> Tensor");
}

TORCH_LIBRARY_IMPL(autosage, CUDA, m) {
  m.impl("spmm_csr", spmm_csr_dispatch);
  m.impl("spmm_csr_split", spmm_csr_split_dispatch);
}
