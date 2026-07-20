cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures")
endif()

project(autosage LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_path(
  AUTOSAGE_TORCH_CUDA_HEADER
  c10/cuda/impl/cuda_cmake_macros.h
  PATHS ${TORCH_INCLUDE_DIRS}
  NO_DEFAULT_PATH
)
if(NOT AUTOSAGE_TORCH_CUDA_HEADER)
  message(FATAL_ERROR "A CUDA-enabled PyTorch installation is required")
endif()

add_library(autosage_cuda SHARED
  csrc/spmm_binding.cpp
  csrc/spmm_csr_vec4.cu
  csrc/launcher_spmm.cu
  csrc/spmm_hub.cu
  csrc/spmm_rows.cu
  csrc/sddmm_csr_op.cu
)

target_include_directories(autosage_cuda PRIVATE include)
target_link_libraries(autosage_cuda PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(autosage_cuda PRIVATE ${TORCH_CXX_FLAGS_LIST})

set_target_properties(autosage_cuda PROPERTIES
  OUTPUT_NAME autosage_cuda
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

install(TARGETS autosage_cuda
  LIBRARY DESTINATION autosage/lib
  RUNTIME DESTINATION autosage/lib
)
