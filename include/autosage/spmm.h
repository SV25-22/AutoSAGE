#pragma once
#include <ATen/ATen.h>
#include <c10/util/Optional.h>

namespace autosage {

// API: CSR (crow, col, optional val) * dense X -> dense Y
at::Tensor spmm_csr_cpu(const at::Tensor& crow,
                        const at::Tensor& col,
                        const c10::optional<at::Tensor>& val,
                        const at::Tensor& X);

at::Tensor spmm_csr_cuda(const at::Tensor& crow,
                         const at::Tensor& col,
                         const c10::optional<at::Tensor>& val,
                         const at::Tensor& X);

} // namespace autosage
