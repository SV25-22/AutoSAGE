#include <ATen/ATen.h>
#include "autosage/spmm.h"

namespace autosage {

// Simple reference CPU impl (float32 only for P1)
at::Tensor spmm_csr_cpu(const at::Tensor& crow,
                        const at::Tensor& col,
                        const c10::optional<at::Tensor>& val,
                        const at::Tensor& X)
{
    TORCH_CHECK(!X.is_cuda(), "spmm_csr_cpu expects CPU tensor X.");
    TORCH_CHECK(crow.scalar_type() == at::kLong && col.scalar_type() == at::kLong,
                "CSR indices must be int64 (torch.long).");
    TORCH_CHECK(X.scalar_type() == at::kFloat, "P1 CPU path supports float32 features only.");
    TORCH_CHECK(crow.dim() == 1 && col.dim() == 1, "CSR crow/col must be 1D.");
    TORCH_CHECK(X.dim() == 2, "X must be [N, F].");

    auto Xc    = X.contiguous();
    auto crowc = crow.contiguous();
    auto colc  = col.contiguous();
    c10::optional<at::Tensor> valc = c10::nullopt;
    if (val.has_value() && val.value().defined()) {
        TORCH_CHECK(val.value().scalar_type() == at::kFloat,
                    "P1 CPU path supports float32 edge weights only.");
        valc = val.value().contiguous();
    }

    const int64_t N   = Xc.size(0);
    const int64_t F   = Xc.size(1);
    const int64_t nnz = colc.size(0);

    TORCH_CHECK(crowc.size(0) == N + 1, "crow length must be N+1 (got %ld vs %ld).",
                (long)crowc.size(0), (long)(N + 1));
    const long last_ptr = crowc.index({N}).item<long>();
    TORCH_CHECK(last_ptr == nnz, "crow[N] must equal nnz=col.size(0) (got %ld vs %ld).",
                last_ptr, (long)nnz);
    if (valc && valc->defined()) {
        TORCH_CHECK(valc->size(0) == nnz, "val length must equal nnz (got %ld vs %ld).",
                    (long)valc->size(0), (long)nnz);
    }

    auto Y = at::zeros_like(Xc);

    const long* crow_p = crowc.data_ptr<long>();
    const long* col_p  = colc.data_ptr<long>();
    const float* val_p = (valc && valc->defined()) ? valc->data_ptr<float>() : nullptr;
    const float* X_p   = Xc.data_ptr<float>();
    float* Y_p         = Y.data_ptr<float>();

    for (int64_t r = 0; r < N; ++r) {
        const long start = crow_p[r];
        const long end   = crow_p[r + 1];
        for (long e = start; e < end; ++e) {
            const int64_t c = static_cast<int64_t>(col_p[e]);
            const float w = val_p ? val_p[e] : 1.0f;
            const float* xrow = X_p + c * F;
            float* yrow = Y_p + r * F;
            for (int64_t f = 0; f < F; ++f) {
                yrow[f] += w * xrow[f];
            }
        }
    }
    return Y;
}

} // namespace autosage
