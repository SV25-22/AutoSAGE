#pragma once
#include <ATen/ATen.h>
#include <c10/util/Optional.h>

namespace autosage {

// Existing public API
at::Tensor spmm_csr_cpu(const at::Tensor& crow,
                        const at::Tensor& col,
                        const c10::optional<at::Tensor>& val,
                        const at::Tensor& X);

at::Tensor spmm_csr_cuda(const at::Tensor& crow,
                         const at::Tensor& col,
                         const c10::optional<at::Tensor>& val,
                         const at::Tensor& X);

// M2: split path (CTA-per-hub + warp-per-row for light rows)
at::Tensor spmm_csr_split_cuda(const at::Tensor& crow,
                               const at::Tensor& col,
                               const c10::optional<at::Tensor>& val,
                               const at::Tensor& X,
                               int ftile, int wpb, int hubT);

// M2: overwrite heavy rows (CTA-per-hub)
void spmm_csr_hub_overwrite_launch(const at::Tensor& crow,
                                   const at::Tensor& col,
                                   const c10::optional<at::Tensor>& val,
                                   const at::Tensor& X, at::Tensor& Y,
                                   const at::Tensor& heavy_rows_i32,
                                   int ftile, int wpb, bool use_vec4);

// M2: overwrite a subset of rows with warp-per-row kernel (LIGHT rows)
void spmm_csr_rows_overwrite_launch(const at::Tensor& crow,
                                    const at::Tensor& col,
                                    const c10::optional<at::Tensor>& val,
                                    const at::Tensor& X, at::Tensor& Y,
                                    const at::Tensor& rows_i32,
                                    int ftile, bool use_vec4);

} // namespace autosage
