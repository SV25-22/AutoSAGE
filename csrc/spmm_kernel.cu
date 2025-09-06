#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include "autosage/spmm.h"

namespace autosage {

at::Tensor spmm_csr_cuda(const at::Tensor& crow,
                         const at::Tensor& col,
                         const c10::optional<at::Tensor>& val,
                         const at::Tensor& X) {
  // P0 stub: just shape-correct zeros so dispatcher / linkage is validated
  auto Y = at::zeros_like(X);
  return Y;
}

} // namespace autosage
