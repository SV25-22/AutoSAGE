#include <cusparse.h>
#include <ATen/ATen.h>

static void check_cusparse(cusparseStatus_t s) {
  TORCH_CHECK(s == CUSPARSE_STATUS_SUCCESS, "cuSPARSE error: ", (int)s);
}

// Example placeholder (not used in P0)
extern "C" void autosage_cusparse_spmm_placeholder() {
  // Intentionally empty; will be implemented when wiring cuSPARSE path.
}
