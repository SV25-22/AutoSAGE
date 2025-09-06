#include <ATen/ATen.h>
#include <torch/library.h>
extern "C" void autosage_spmm_vec4_launch(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, at::Tensor&);

#include <c10/util/Optional.h>
#include "autosage/spmm.h"

namespace {
using c10::optional;

static at::Tensor spmm_csr_dispatch(const at::Tensor& crow,
                                    const at::Tensor& col,
                                    const optional<at::Tensor>& val,
                                    const at::Tensor& X) {
  TORCH_CHECK(crow.is_contiguous() && col.is_contiguous(), "CSR arrays must be contiguous");
  TORCH_CHECK(crow.dtype() == at::kLong && col.dtype() == at::kLong, "CSR indices must be int64");
  TORCH_CHECK(X.is_contiguous(), "X must be contiguous");

  if (X.is_cuda()) {
    return autosage::spmm_csr_cuda(crow, col, val, X);
  } else {
    return autosage::spmm_csr_cpu(crow, col, val, X);
  }
}
} // anon

TORCH_LIBRARY(autosage, m) {
  m.def("spmm_csr(Tensor crow, Tensor col, Tensor? val, Tensor X) -> Tensor");
}

TORCH_LIBRARY_IMPL(autosage, CPU, m) {
  m.impl("spmm_csr", spmm_csr_dispatch);
}

TORCH_LIBRARY_IMPL(autosage, CUDA, m) {
  m.impl("spmm_csr", spmm_csr_dispatch);
}
