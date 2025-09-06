#pragma once
#include <ATen/ATen.h>
#include <cstdint>

namespace autosage {

// Layout: [N, nnz, mean_deg, max_deg, q50, q90, q99, heavy_tail]
at::Tensor compute_graph_stats(const at::Tensor& crow, const at::Tensor& col);

// Layout: [sm_count, cc_major, cc_minor, l2_bytes, shared_per_sm, regs_per_sm, warp_size]
at::Tensor get_device_info(int64_t device_index);

} // namespace autosage
