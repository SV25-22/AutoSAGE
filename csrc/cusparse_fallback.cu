#include <ATen/ATen.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdexcept>

static void throw_if_cusparse(cusparseStatus_t st, const char* msg){
  if (st != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(msg);
  }
}

// autosage::cusparse_spmm_csr(Tensor crow, Tensor col, Tensor? val, Tensor x) -> Tensor
static at::Tensor cusparse_spmm_csr_kernel(const at::Tensor& crow, const at::Tensor& col,
                                           const c10::optional<at::Tensor>& val_opt,
                                           const at::Tensor& X) {
#ifndef AUTOSAGE_WITH_CUSPARSE
  TORCH_CHECK(false, "Built without cuSPARSE");
#else
  TORCH_CHECK(crow.is_cuda() && col.is_cuda() && X.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(crow.dtype()==at::kLong && col.dtype()==at::kLong, "crow/col must be int64");
  TORCH_CHECK(X.dtype()==at::kFloat, "X must be float32");
  const int64_t N = X.size(0), F = X.size(1);
  at::Tensor Y = at::zeros({N, F}, X.options());

  cusparseHandle_t handle; throw_if_cusparse(cusparseCreate(&handle), "cusparseCreate");
  // descriptors
  auto crow_i32 = crow.to(at::kInt);  // cusparse needs int32
  auto col_i32  = col.to(at::kInt);
  const int64_t nnz = col.size(0);
  const float alpha = 1.f, beta = 0.f;
  cusparseSpMatDescr_t A;
  if (val_opt.has_value() && val_opt->defined()) {
    auto V = val_opt->to(at::kFloat);
    throw_if_cusparse(cusparseCreateCsr(&A, (int)N, (int)N, (int)nnz,
                                        crow_i32.data_ptr<int>(), col_i32.data_ptr<int>(),
                                        V.data_ptr<float>(),
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
                      "cusparseCreateCsr (weighted)");
  } else {
    // treat values as all 1s using a dummy dense vector multiply trick via gather? Simplest: actually pass ones
    auto V = at::ones({nnz}, X.options());
    throw_if_cusparse(cusparseCreateCsr(&A, (int)N, (int)N, (int)nnz,
                                        crow_i32.data_ptr<int>(), col_i32.data_ptr<int>(),
                                        V.data_ptr<float>(),
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
                      "cusparseCreateCsr (unweighted)");
  }
  cusparseDnMatDescr_t B, C;
  throw_if_cusparse(cusparseCreateDnMat(&B, (int)N, (int)F, (int)F, const_cast<float*>(X.data_ptr<float>()),
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW), "createDnMat B");
  throw_if_cusparse(cusparseCreateDnMat(&C, (int)N, (int)F, (int)F, Y.data_ptr<float>(),
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW), "createDnMat C");
  // buffer
  size_t bufSize=0; void* dBuf=nullptr;
  throw_if_cusparse(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, A, B, &beta, C, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufSize), "SpMM_bufferSize");
  if (bufSize) cudaMalloc(&dBuf, bufSize);
  throw_if_cusparse(cusparseSpMM(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, A, B, &beta, C, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuf), "SpMM");
  if (dBuf) cudaFree(dBuf);
  cusparseDestroySpMat(A); cusparseDestroyDnMat(B); cusparseDestroyDnMat(C);
  cusparseDestroy(handle);
  return Y;
#endif
}

// register only if compiled with the flag
TORCH_LIBRARY_FRAGMENT(autosage, m) {
  m.def("cusparse_spmm_csr(Tensor crow, Tensor col, Tensor? val, Tensor x) -> Tensor");
}
TORCH_LIBRARY_IMPL(autosage, CUDA, m) {
  m.impl("cusparse_spmm_csr", cusparse_spmm_csr_kernel);
}
