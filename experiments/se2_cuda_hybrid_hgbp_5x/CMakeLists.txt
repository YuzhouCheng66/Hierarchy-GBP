cmake_minimum_required(VERSION 3.24)

project(se2_cuda_hybrid_hgbp LANGUAGES CXX CUDA)

if(NOT WIN32)
    message(FATAL_ERROR "This recovered prototype uses LoadLibrary for CHOLMOD and currently supports Windows only.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Ada / RTX 4080, matching the machine used for the archived measurement.
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

set(EIGEN3_INCLUDE_DIR "" CACHE PATH "Directory containing Eigen/Dense")
set(SUITESPARSE_INCLUDE_DIR "" CACHE PATH "Directory containing cholmod.h")

if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Dense")
    message(FATAL_ERROR "Set EIGEN3_INCLUDE_DIR to the Eigen include root.")
endif()
if(NOT EXISTS "${SUITESPARSE_INCLUDE_DIR}/cholmod.h")
    message(FATAL_ERROR "Set SUITESPARSE_INCLUDE_DIR to the directory containing cholmod.h.")
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(se2_cuda_hybrid_hgbp
    se2_cuda_hybrid_hgbp.cu
)

target_include_directories(se2_cuda_hybrid_hgbp PRIVATE
    "${EIGEN3_INCLUDE_DIR}"
    "${SUITESPARSE_INCLUDE_DIR}"
)

target_compile_options(se2_cuda_hybrid_hgbp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--maxrregcount=152;-Xcompiler=/openmp>
)

target_link_libraries(se2_cuda_hybrid_hgbp PRIVATE
    CUDA::cudart
    CUDA::cusolver
    CUDA::cublas
)

set_target_properties(se2_cuda_hybrid_hgbp PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
