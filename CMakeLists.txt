cmake_minimum_required(VERSION 3.18)
project(autosage LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# PyTorch (pass -DCMAKE_PREFIX_PATH from your venv)
find_package(Torch REQUIRED)

# CUDA archs (override with -DCMAKE_CUDA_ARCHITECTURES=...)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

# cuSPARSE fallback OFF by default (NVCC 11.x template quirks)
option(AUTOSAGE_WITH_CUSPARSE "Build cuSPARSE fallback" OFF)

set(AUTOSAGE_SRC
  csrc/spmm_binding.cpp
  csrc/spmm_cpu.cpp            # ok if present; else remove this line
  csrc/spmm_csr_vec4.cu
  csrc/stats.cpp
  csrc/scheduler.cpp
  csrc/launcher_spmm.cu
  csrc/spmm_hub.cu 
  csrc/spmm_rows.cu
)

if (AUTOSAGE_WITH_CUSPARSE)
  list(APPEND AUTOSAGE_SRC csrc/cusparse_fallback.cu)
  add_definitions(-DAUTOSAGE_WITH_CUSPARSE=1)
endif()

add_library(autosage_cuda SHARED ${AUTOSAGE_SRC})
target_include_directories(autosage_cuda PRIVATE ${TORCH_INCLUDE_DIRS} include)
target_link_libraries(autosage_cuda PRIVATE ${TORCH_LIBRARIES} c10)

# Drop the .so into cmake-build/
set_target_properties(autosage_cuda PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
