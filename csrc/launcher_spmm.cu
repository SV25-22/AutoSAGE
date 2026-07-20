#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include "autosage/spmm.h"

namespace autosage {
namespace {

inline at::Tensor row_degrees(const at::Tensor& crow) {
  const auto n = crow.size(0);
  auto next = crow.slice(0, 1, n);
  auto curr = crow.slice(0, 0, n - 1);
  return next - curr;
}

} 

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

  bool vec4_enabled = true;
  if (const char* value = std::getenv("AUTOSAGE_VEC4")) {
    std::string setting(value);
    std::transform(setting.begin(), setting.end(), setting.begin(),
                   [](unsigned char character) {
                     return static_cast<char>(std::tolower(character));
                   });
    vec4_enabled = setting != "0" && setting != "false" && setting != "no" && setting != "off";
  }
  const bool vec4_ok =
    (vec4_enabled && X.scalar_type()==at::kFloat &&
     (X.size(1) % 4 == 0) &&
     (reinterpret_cast<uintptr_t>(X.data_ptr<float>()) % 16 == 0));

  at::Tensor heavy_rows_i32, light_rows_i32;
  {
    auto deg = row_degrees(crow);                
    at::Tensor heavy_mask = (hubT > 0) ? deg.ge(hubT) : at::zeros_like(deg, deg.options().dtype(at::kBool));
    at::Tensor light_mask = heavy_mask.logical_not();

    auto heavy_rows = at::nonzero(heavy_mask).flatten(); 
    auto light_rows = at::nonzero(light_mask).flatten(); 
    heavy_rows_i32 = heavy_rows.toType(at::kInt).contiguous();
    light_rows_i32 = light_rows.toType(at::kInt).contiguous();
  }

  at::Tensor Y = at::empty_like(X);

  if (heavy_rows_i32.numel() > 0) {
    spmm_csr_hub_overwrite_launch(crow, col, val, X, Y, heavy_rows_i32, ftile, wpb, vec4_ok);
  }

  if (light_rows_i32.numel() > 0) {
    spmm_csr_rows_overwrite_launch(crow, col, val, X, Y, light_rows_i32, ftile, vec4_ok);
  }

  return Y;
}

} 
