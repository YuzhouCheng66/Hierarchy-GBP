cmake_minimum_required(VERSION 3.20)

# Keep compiler sanity checks on the same runtime as the benchmark build.
set(CMAKE_TRY_COMPILE_CONFIGURATION Release)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

project(HierarchyGBP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(HGBP_ENABLE_ASAN "Build with MSVC AddressSanitizer for memory diagnostics." OFF)

set(HGBP_DEPENDENCY_ROOT "" CACHE PATH
    "Dependency workspace containing external/eigen, vcpkg_installed/x64-windows, and external_baselines/rootba.")
if(NOT HGBP_DEPENDENCY_ROOT AND DEFINED ENV{HGBP_DEPENDENCY_ROOT})
    set(HGBP_DEPENDENCY_ROOT "$ENV{HGBP_DEPENDENCY_ROOT}" CACHE PATH
        "Dependency workspace containing the benchmark dependencies." FORCE)
endif()

set(_hgbp_eigen_default "")
set(_hgbp_vcpkg_default "")
set(_hgbp_rootba_default "")
if(HGBP_DEPENDENCY_ROOT)
    set(_hgbp_eigen_default "${HGBP_DEPENDENCY_ROOT}/external/eigen")
    set(_hgbp_vcpkg_default "${HGBP_DEPENDENCY_ROOT}/vcpkg_installed/x64-windows")
    set(_hgbp_rootba_default "${HGBP_DEPENDENCY_ROOT}/external_baselines/rootba")
endif()
if(DEFINED ENV{HGBP_EIGEN_DIR})
    set(_hgbp_eigen_default "$ENV{HGBP_EIGEN_DIR}")
endif()
if(DEFINED ENV{HGBP_VCPKG_INSTALLED_DIR})
    set(_hgbp_vcpkg_default "$ENV{HGBP_VCPKG_INSTALLED_DIR}")
endif()
if(DEFINED ENV{HGBP_ROOTBA_DIR})
    set(_hgbp_rootba_default "$ENV{HGBP_ROOTBA_DIR}")
endif()

set(HGBP_EIGEN_DIR "${_hgbp_eigen_default}" CACHE PATH "Eigen include directory.")
set(HGBP_VCPKG_INSTALLED_DIR "${_hgbp_vcpkg_default}" CACHE PATH
    "SuiteSparse/OpenBLAS vcpkg installation directory.")
set(HGBP_ROOTBA_DIR "${_hgbp_rootba_default}" CACHE PATH
    "RootBA checkout used by the optimized BA linearization.")

foreach(_dependency HGBP_EIGEN_DIR HGBP_VCPKG_INSTALLED_DIR HGBP_ROOTBA_DIR)
    if(NOT IS_DIRECTORY "${${_dependency}}")
        message(FATAL_ERROR
            "${_dependency}='${${_dependency}}' is not a directory. "
            "Set HGBP_DEPENDENCY_ROOT or pass the three dependency paths explicitly.")
    endif()
endforeach()

find_package(OpenMP REQUIRED)

add_library(hgbp_se3
    src/gaussian.cpp
    src/factor.cpp
    src/variable.cpp
    src/factor_graph.cpp
    src/se3_internal_residual.cpp
    src/se3_solver.cpp
)

target_include_directories(hgbp_se3
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${HGBP_EIGEN_DIR}
        ${HGBP_VCPKG_INSTALLED_DIR}/include
        ${HGBP_VCPKG_INSTALLED_DIR}/include/suitesparse
)

target_compile_definitions(hgbp_se3 PUBLIC EIGEN_NO_DEBUG)
target_link_directories(hgbp_se3 PUBLIC "${HGBP_VCPKG_INSTALLED_DIR}/lib")

target_link_libraries(hgbp_se3
    PUBLIC
        OpenMP::OpenMP_CXX
    PRIVATE
        cholmod
        suitesparseconfig
        amd
        colamd
        camd
        ccolamd
        lapack
        openblas
)

if(MSVC)
    target_compile_options(hgbp_se3 PRIVATE
        /bigobj
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/Ob2>
        $<$<CONFIG:Release>:/Gy>
        $<$<CONFIG:Release>:/Gw>
        $<$<CONFIG:Release>:/arch:AVX2>
        $<$<CONFIG:Release>:/DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/Ob2>
        $<$<CONFIG:RelWithDebInfo>:/Gy>
        $<$<CONFIG:RelWithDebInfo>:/Gw>
        $<$<CONFIG:RelWithDebInfo>:/arch:AVX2>
        $<$<CONFIG:RelWithDebInfo>:/DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:/Zi>
        $<$<CONFIG:RelWithDebInfo>:/Zo>
        $<$<CONFIG:RelWithDebInfo>:/Oy->
    )
    target_link_options(hgbp_se3 PRIVATE
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>
        $<$<CONFIG:Release>:/INCREMENTAL:NO>
        $<$<CONFIG:RelWithDebInfo>:/OPT:REF>
        $<$<CONFIG:RelWithDebInfo>:/OPT:ICF>
        $<$<CONFIG:RelWithDebInfo>:/INCREMENTAL:NO>
        $<$<CONFIG:RelWithDebInfo>:/DEBUG:FASTLINK>
    )
else()
    target_compile_options(hgbp_se3 PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-g>
    )
endif()

add_library(hgbp_se2
    src/gaussian.cpp
    src/factor.cpp
    src/variable.cpp
    src/factor_graph.cpp
    src/se2_internal_sync.cpp
    src/se2_internal_group_sync.cpp
    src/se2_internal_dense_sync.cpp
    src/se2_internal_basis.cpp
    src/se2_internal_coarse.cpp
    src/se2_internal_residual.cpp
    src/se2_solver.cpp
)

target_include_directories(hgbp_se2
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${HGBP_EIGEN_DIR}
        ${HGBP_VCPKG_INSTALLED_DIR}/include
        ${HGBP_VCPKG_INSTALLED_DIR}/include/suitesparse
)

target_compile_definitions(hgbp_se2 PUBLIC EIGEN_NO_DEBUG)
target_link_directories(hgbp_se2 PUBLIC "${HGBP_VCPKG_INSTALLED_DIR}/lib")

target_link_libraries(hgbp_se2
    PUBLIC
        OpenMP::OpenMP_CXX
    PRIVATE
        cholmod
        suitesparseconfig
        amd
        colamd
        camd
        ccolamd
        lapack
        openblas
)

if(MSVC)
    set_source_files_properties(
        src/se2_internal_residual.cpp
        PROPERTIES COMPILE_OPTIONS "/fp:fast"
    )
    target_compile_options(hgbp_se2 PRIVATE
        /bigobj
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/Ob2>
        $<$<CONFIG:Release>:/Gy>
        $<$<CONFIG:Release>:/Gw>
        $<$<CONFIG:Release>:/arch:AVX2>
        $<$<CONFIG:Release>:/DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/Ob2>
        $<$<CONFIG:RelWithDebInfo>:/Gy>
        $<$<CONFIG:RelWithDebInfo>:/Gw>
        $<$<CONFIG:RelWithDebInfo>:/arch:AVX2>
        $<$<CONFIG:RelWithDebInfo>:/DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:/Zi>
        $<$<CONFIG:RelWithDebInfo>:/Zo>
        $<$<CONFIG:RelWithDebInfo>:/Oy->
    )
    target_link_options(hgbp_se2 PRIVATE
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>
        $<$<CONFIG:Release>:/INCREMENTAL:NO>
        $<$<CONFIG:RelWithDebInfo>:/OPT:REF>
        $<$<CONFIG:RelWithDebInfo>:/OPT:ICF>
        $<$<CONFIG:RelWithDebInfo>:/INCREMENTAL:NO>
        $<$<CONFIG:RelWithDebInfo>:/DEBUG:FASTLINK>
    )
else()
    target_compile_options(hgbp_se2 PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-g>
    )
endif()

add_executable(se3_benchmark src/se3_benchmark.cpp)
target_link_libraries(se3_benchmark PRIVATE hgbp_se3)

add_executable(se2_benchmark src/se2_benchmark.cpp)
target_link_libraries(se2_benchmark PRIVATE hgbp_se2)

add_executable(ba_solver src/ba_solver.cpp)
target_include_directories(ba_solver PRIVATE
    ${HGBP_EIGEN_DIR}
    ${HGBP_VCPKG_INSTALLED_DIR}/include
    ${HGBP_VCPKG_INSTALLED_DIR}/include/suitesparse
    ${HGBP_ROOTBA_DIR}/src
    ${HGBP_ROOTBA_DIR}/external/Sophus
    ${HGBP_ROOTBA_DIR}/external/visit_struct/include
    ${HGBP_ROOTBA_DIR}/external/wise_enum
    ${HGBP_ROOTBA_DIR}/external/enum-flags/include
    ${HGBP_ROOTBA_DIR}/external/nameof/include
    ${HGBP_ROOTBA_DIR}/external/toml11
    ${HGBP_ROOTBA_DIR}/external/json
    ${HGBP_ROOTBA_DIR}/external/basalt-headers/include
    ${HGBP_ROOTBA_DIR}/external/magic_enum
    ${HGBP_ROOTBA_DIR}/external/pprint
    ${HGBP_ROOTBA_DIR}/external/cereal/include
    ${HGBP_ROOTBA_DIR}/conda_env/Library/include
)
target_link_directories(ba_solver PRIVATE
    ${HGBP_VCPKG_INSTALLED_DIR}/lib
    ${HGBP_ROOTBA_DIR}/build-win/src/rootba/Release
    ${HGBP_ROOTBA_DIR}/conda_env/Library/lib
)
target_compile_definitions(ba_solver PRIVATE
    ROOTBA_DISABLE_ASSERTS
    ROOTBA_INSTANTIATIONS_DOUBLE
    ROOTBA_INSTANTIATIONS_STATIC_LMB
    CEREAL_THREAD_SAFE=1
    FMT_SHARED
    GLOG_USE_GLOG_EXPORT
    GLOG_USE_GFLAGS
    GLOG_NO_ABBREVIATED_SEVERITIES
    GFLAGS_IS_A_DLL=1
)
target_link_libraries(ba_solver PRIVATE
    OpenMP::OpenMP_CXX
    rootba
    tbb12
    glog
    gflags
    fmt
    cholmod
    suitesparseconfig
    amd
    colamd
    camd
    ccolamd
    lapack
    openblas
)

if(MSVC)
    target_compile_options(se3_benchmark PRIVATE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/Ob2>
        $<$<CONFIG:Release>:/Gy>
        $<$<CONFIG:Release>:/Gw>
        $<$<CONFIG:Release>:/arch:AVX2>
        $<$<CONFIG:Release>:/DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/Ob2>
        $<$<CONFIG:RelWithDebInfo>:/Gy>
        $<$<CONFIG:RelWithDebInfo>:/Gw>
        $<$<CONFIG:RelWithDebInfo>:/arch:AVX2>
        $<$<CONFIG:RelWithDebInfo>:/DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:/Zi>
        $<$<CONFIG:RelWithDebInfo>:/Zo>
        $<$<CONFIG:RelWithDebInfo>:/Oy->
    )
    target_link_options(se3_benchmark PRIVATE
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>
        $<$<CONFIG:Release>:/INCREMENTAL:NO>
        $<$<CONFIG:RelWithDebInfo>:/OPT:REF>
        $<$<CONFIG:RelWithDebInfo>:/OPT:ICF>
        $<$<CONFIG:RelWithDebInfo>:/INCREMENTAL:NO>
        $<$<CONFIG:RelWithDebInfo>:/DEBUG:FASTLINK>
    )
    target_compile_options(se2_benchmark PRIVATE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/Ob2>
        $<$<CONFIG:Release>:/Gy>
        $<$<CONFIG:Release>:/Gw>
        $<$<CONFIG:Release>:/arch:AVX2>
        $<$<CONFIG:Release>:/DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/Ob2>
        $<$<CONFIG:RelWithDebInfo>:/Gy>
        $<$<CONFIG:RelWithDebInfo>:/Gw>
        $<$<CONFIG:RelWithDebInfo>:/arch:AVX2>
        $<$<CONFIG:RelWithDebInfo>:/DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:/Zi>
        $<$<CONFIG:RelWithDebInfo>:/Zo>
        $<$<CONFIG:RelWithDebInfo>:/Oy->
    )
    target_link_options(se2_benchmark PRIVATE
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>
        $<$<CONFIG:Release>:/INCREMENTAL:NO>
        $<$<CONFIG:RelWithDebInfo>:/OPT:REF>
        $<$<CONFIG:RelWithDebInfo>:/OPT:ICF>
        $<$<CONFIG:RelWithDebInfo>:/INCREMENTAL:NO>
        $<$<CONFIG:RelWithDebInfo>:/DEBUG:FASTLINK>
    )
    target_compile_options(ba_solver PRIVATE
        /bigobj
        /Zc:__cplusplus
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/Ob2>
        $<$<CONFIG:Release>:/Gy>
        $<$<CONFIG:Release>:/Gw>
        $<$<CONFIG:Release>:/arch:AVX2>
        $<$<CONFIG:Release>:/DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/Ob2>
        $<$<CONFIG:RelWithDebInfo>:/Gy>
        $<$<CONFIG:RelWithDebInfo>:/Gw>
        $<$<CONFIG:RelWithDebInfo>:/arch:AVX2>
        $<$<CONFIG:RelWithDebInfo>:/DNDEBUG>
    )
    target_link_options(ba_solver PRIVATE
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>
        $<$<CONFIG:Release>:/INCREMENTAL:NO>
        $<$<CONFIG:RelWithDebInfo>:/OPT:REF>
        $<$<CONFIG:RelWithDebInfo>:/OPT:ICF>
        $<$<CONFIG:RelWithDebInfo>:/INCREMENTAL:NO>
    )
endif()

if(MSVC)
    foreach(_target hgbp_se3 hgbp_se2 se3_benchmark se2_benchmark ba_solver)
        target_compile_options(${_target} PRIVATE /wd4392 /w14505)
    endforeach()
endif()

if(HGBP_ENABLE_ASAN AND MSVC)
    foreach(_target hgbp_se3 hgbp_se2 se3_benchmark se2_benchmark ba_solver)
        target_compile_options(${_target} PRIVATE /fsanitize=address /Zi /Od)
        target_link_options(${_target} PRIVATE /INCREMENTAL:NO)
    endforeach()
endif()
