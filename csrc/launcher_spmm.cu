#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include "autosage/spmm.h"

namespace autosage {
namespace {

// degrees = crow[1:] - crow[:-1]
inline at::Tensor row_degrees(const at::Tensor& crow) {
  const auto n = crow.size(0);
  auto next = crow.slice(0, 1, n);
  auto curr = crow.slice(0, 0, n - 1);
  return next - curr;
}

} // anon

at::Tensor spmm_csr_split_cuda(
    const at::Tensor& crow,
    const at::Tensor& col,
    const c10::optional<at::Tensor>& val,
    const at::Tensor& X,
    int ftile, int wpb, int hubT)
{
  TORCH_CHECK(crow.is_cuda() && col.is_cuda() && X.is_cuda(),
              "crow/col/X must be CUDA tensors");
  TORCH_CHECK(crow.scalar_type()==at::kLong && col.scalar_type()==at::kLong,
              "CSR indices must be int64 (torch.long)");
  TORCH_CHECK(X.dim()==2 && X.is_contiguous(), "X must be (N,F) contiguous");

  const bool vec4_ok =
    (X.scalar_type()==at::kFloat &&
     (X.size(1) % 4 == 0) &&
     (reinterpret_cast<uintptr_t>(X.data_ptr<float>()) % 16 == 0));

  // Build heavy/light row partitions
  at::Tensor heavy_rows_i32, light_rows_i32;
  {
    auto deg = row_degrees(crow);                 // [N] int64
    at::Tensor heavy_mask = (hubT > 0) ? deg.ge(hubT) : at::zeros_like(deg, deg.options().dtype(at::kBool));
    at::Tensor light_mask = heavy_mask.logical_not();

    auto heavy_rows = at::nonzero(heavy_mask).flatten(); // [H] int64
    auto light_rows = at::nonzero(light_mask).flatten(); // [L] int64
    heavy_rows_i32 = heavy_rows.toType(at::kInt).contiguous();
    light_rows_i32 = light_rows.toType(at::kInt).contiguous();
  }

  // Prepare output
  at::Tensor Y = at::empty_like(X);

  // 1) CTA-per-hub: overwrite heavy rows in Y
  if (heavy_rows_i32.numel() > 0) {
    spmm_csr_hub_overwrite_launch(crow, col, val, X, Y, heavy_rows_i32, ftile, wpb, vec4_ok);
  }

  // 2) Warp-per-row on LIGHT rows only (overwrites those rows in Y)
  if (light_rows_i32.numel() > 0) {
    spmm_csr_rows_overwrite_launch(crow, col, val, X, Y, light_rows_i32, ftile, vec4_ok);
  }

  return Y;
}

} // namespace autosage
