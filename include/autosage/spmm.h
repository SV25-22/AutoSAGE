#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

namespace autosage {

at::Tensor spmm_csr_cuda(
    const at::Tensor& crow,
    const at::Tensor& col,
    const c10::optional<at::Tensor>& val,
    const at::Tensor& x);

at::Tensor spmm_csr_split_cuda(
    const at::Tensor& crow,
    const at::Tensor& col,
    const c10::optional<at::Tensor>& val,
    const at::Tensor& x,
    int feature_tile,
    int warps_per_block,
    int hub_threshold);

void spmm_csr_hub_overwrite_launch(
    const at::Tensor& crow,
    const at::Tensor& col,
    const c10::optional<at::Tensor>& val,
    const at::Tensor& x,
    at::Tensor& output,
    const at::Tensor& heavy_rows,
    int feature_tile,
    int warps_per_block,
    bool use_vec4);

void spmm_csr_rows_overwrite_launch(
    const at::Tensor& crow,
    const at::Tensor& col,
    const c10::optional<at::Tensor>& val,
    const at::Tensor& x,
    at::Tensor& output,
    const at::Tensor& rows,
    int feature_tile,
    bool use_vec4);

}
