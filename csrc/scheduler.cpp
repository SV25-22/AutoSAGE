#include <ATen/ATen.h>
#include <torch/library.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace autosage {
  at::Tensor compute_graph_stats(const at::Tensor& crow, const at::Tensor& col);
}

namespace {

struct Cand { int64_t ftile; int64_t wpb; int64_t hubT; double score; };

static inline double pick_target_wpb(double heavy_tail) {
  if (heavy_tail >= 4.0) return 8.0;
  if (heavy_tail >= 2.0) return 4.0;
  return 2.0;
}

static std::vector<Cand> enumerate_and_score(int64_t F, double heavy_tail) {
  const int64_t ftiles[] = {32, 64, 128};
  const int64_t wpbs[]   = {2, 4, 8};
  const double  target   = pick_target_wpb(heavy_tail);

  std::vector<Cand> out; out.reserve(9);
  for (auto ft : ftiles) for (auto wb : wpbs) {
    double s = 0.0;
    // Prefer larger tiles up to F
    s += 1.0 / static_cast<double>(std::min<int64_t>(ft, F));
    // Mild penalty if tile doesn't divide F
    if (F % ft != 0) s += 0.15;
    // Encourage wpb close to target (log-space distance)
    s += 0.03 * std::abs(std::log2(static_cast<double>(wb)) - std::log2(target));
    out.push_back({ft, wb, 0, s}); // hubT reserved for P1.5
  }
  std::sort(out.begin(), out.end(), [](const Cand& a, const Cand& b){ return a.score < b.score; });
  return out;
}

at::Tensor model_topk_kernel(const at::Tensor& crow, const at::Tensor& col, int64_t F, int64_t k) {
  TORCH_CHECK(crow.dtype()==at::kLong && col.dtype()==at::kLong, "crow/col must be int64");
  TORCH_CHECK(crow.dim()==1 && col.dim()==1, "crow/col must be 1-D");
  TORCH_CHECK(k > 0, "k must be > 0");

  // Graph stats: [N, nnz, mean, max, q50, q90, q99, heavy]
  at::Tensor stats = autosage::compute_graph_stats(crow, col);
  const float heavy_f = stats.data_ptr<float>()[7];
  const double heavy  = static_cast<double>(heavy_f);

  auto ranked = enumerate_and_score(F, heavy);
  if (static_cast<size_t>(k) > ranked.size()) k = static_cast<int64_t>(ranked.size());

  at::Tensor out = at::empty({k, 3}, at::device(at::kCPU).dtype(at::kLong));
  auto* p = out.data_ptr<int64_t>();
  for (int64_t i = 0; i < k; ++i) {
    p[3*i+0] = ranked[i].ftile;
    p[3*i+1] = ranked[i].wpb;
    p[3*i+2] = ranked[i].hubT;
  }
  return out;
}

} // namespace

// Schema (fragment to avoid duplicate TORCH_LIBRARY)
TORCH_LIBRARY_FRAGMENT(autosage, m) {
  m.def("model_topk(Tensor crow, Tensor col, int F, int k) -> Tensor");
}

// Register for CPU & CUDA (has Tensor inputs so normal dispatch works)
TORCH_LIBRARY_IMPL(autosage, CPU, m)  { m.impl("model_topk", TORCH_FN(model_topk_kernel)); }
TORCH_LIBRARY_IMPL(autosage, CUDA, m) { m.impl("model_topk", TORCH_FN(model_topk_kernel)); }
