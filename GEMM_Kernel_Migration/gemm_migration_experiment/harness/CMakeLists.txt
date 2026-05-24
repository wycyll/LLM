cmake_minimum_required(VERSION 3.24)
project(gemm_migration_harness LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED KERNEL_SOURCE)
  message(FATAL_ERROR "Set -DKERNEL_SOURCE=/absolute/path/to/kernel_extracted.cu")
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(gemm_runner
  runner/main.cpp
  runner/cublas_reference.cu
  ${KERNEL_SOURCE}
)

set_source_files_properties(runner/main.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(gemm_runner PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gemm_runner PRIVATE
  CUDA::cudart
  CUDA::cublas
)

target_compile_options(gemm_runner PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr>
)

set_target_properties(gemm_runner PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
