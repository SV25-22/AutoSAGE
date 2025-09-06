cmake_minimum_required(VERSION 3.18)
project(autosage LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Torch (use the one from your venv via CMAKE_PREFIX_PATH in the script)
find_package(Torch REQUIRED)  # provided by pip torch: .../torch/share/cmake
include(cmake/TorchSetup.cmake)  # adds CUDA compile opts (see below)

# ---- CUDA archs (safe defaults; override with -DCMAKE_CUDA_ARCHITECTURES=...)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()


# ---- Sources (cuSPARSE stub OFF by default to avoid nvcc 11.x template quirks)
set(AUTOSAGE_SRC
  csrc/spmm_binding.cpp
  csrc/spmm_cpu.cpp
  csrc/spmm_kernel.cu     # enabled for P1
  csrc/spmm_vec4.cu
  # csrc/cusparse_fallback.cu will be added by flag
  csrc/spmm_vec4.cu
  # csrc/cusparse_fallback.cu will be added by flag
  csrc/stats.cpp
  csrc/scheduler.cpp
  # csrc/cusparse_fallback.cu  # keep OFF for now
)


option(AUTOSAGE_WITH_CUSPARSE "Build cuSPARSE fallback (needs clean CUDA 12.x toolchain)" ON)
if (AUTOSAGE_WITH_CUSPARSE)
  list(APPEND AUTOSAGE_SRC csrc/cusparse_fallback.cu)
  add_definitions(-DAUTOSAGE_WITH_CUSPARSE=1)
endif()

add_library(autosage_cuda SHARED ${AUTOSAGE_SRC})
target_include_directories(autosage_cuda PRIVATE ${TORCH_INCLUDE_DIRS} include)
target_link_libraries(autosage_cuda PRIVATE ${TORCH_LIBRARIES} c10)


# Properties (Torch-style extension)
set_property(TARGET autosage_cuda PROPERTY CXX_STANDARD 17)
set_property(TARGET autosage_cuda PROPERTY CUDA_STANDARD 17)

# macOS dylib suffix (Linux .so is default)
if(APPLE)
  set_target_properties(autosage_cuda PROPERTIES SUFFIX ".dylib")
endif()

# If the option is ON, add the source and define
if (AUTOSAGE_WITH_CUSPARSE)
  list(APPEND AUTOSAGE_SRC csrc/cusparse_fallback.cu)
  add_definitions(-DAUTOSAGE_WITH_CUSPARSE=1)
endif()
