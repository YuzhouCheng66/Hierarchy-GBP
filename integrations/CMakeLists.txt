cmake_minimum_required(VERSION 3.20)
project(HGBPFileAdapters LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
option(HGBP_CAPTURE_POWERBA "Build optional official PowerBA visualization observer" OFF)
get_filename_component(HGBP_PROJECT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
add_subdirectory("${HGBP_PROJECT}/code" "${CMAKE_CURRENT_BINARY_DIR}/core" EXCLUDE_FROM_ALL)

# Generate a diagnostic entry from the maintained source. Fail rather than
# silently patch a different solver. The production source is never rewritten.
file(READ "${HGBP_PROJECT}/code/src/ba_solver.cpp" _source)
foreach(_marker "double lambda = lab.initial_lambda;" "result.eta_sweeps.push_back(stats.eta_sweeps);" "    return result;\n}")
  string(FIND "${_source}" "${_marker}" _position)
  if(_position LESS 0)
    message(FATAL_ERROR "BA adapter hook no longer matches the maintained entry point")
  endif()
endforeach()
string(REPLACE "double lambda = lab.initial_lambda;" "hgbp_io::snapshot(problem, 0);\n    double lambda = lab.initial_lambda;" _source "${_source}")
string(REPLACE "result.eta_sweeps.push_back(stats.eta_sweeps);" "result.eta_sweeps.push_back(stats.eta_sweeps);\n        hgbp_io::snapshot(problem, outer);" _source "${_source}")
string(REPLACE "    return result;\n}" "    hgbp_io::finish(problem);\n    return result;\n}" _source "${_source}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ba_file.cpp" "#include \"ba_io.h\"\n${_source}")
add_executable(ba_file "${CMAKE_CURRENT_BINARY_DIR}/ba_file.cpp")
target_include_directories(ba_file PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${HGBP_PROJECT}/code/src")
foreach(_dependency tbb12 glog gflags fmt)
  list(APPEND _rootba_runtime_libs "${HGBP_ROOTBA_PREFIX}/lib/${_dependency}.lib")
endforeach()
target_link_libraries(ba_file PRIVATE ${_rootba_runtime_libs} hgbp_ba_dependencies)
hgbp_compile(ba_file)
if(MSVC)
  target_compile_options(ba_file PRIVATE /Zc:__cplusplus)
endif()

if(HGBP_CAPTURE_POWERBA)
  set(HGBP_ABSL_PREFIX "" CACHE PATH "Matching external Abseil prefix for the pinned RootBA build")
  file(GLOB _absl_libs "${HGBP_ABSL_PREFIX}/lib/absl*.lib")
  if(NOT _absl_libs)
    message(FATAL_ERROR "Set HGBP_ABSL_PREFIX for the optional PowerBA observer")
  endif()
  find_file(_clipp_header clipp.h PATHS "${HGBP_ROOTBA_DIR}/external/clipp/clipp/include"
    "${HGBP_ROOTBA_DIR}/external/clipp/include" NO_DEFAULT_PATH REQUIRED)
  configure_file("${_clipp_header}" "${CMAKE_CURRENT_BINARY_DIR}/include/clipp/clipp.h" COPYONLY)
  add_executable(powerba_capture ../site/tools/powerba_capture.cpp
    "${HGBP_ROOTBA_DIR}/src/rootba/solver/bal_bundle_adjustment.cpp"
    "${HGBP_ROOTBA_DIR}/src/rootba/solver/linearizor.cpp"
    "${HGBP_ROOTBA_DIR}/src/rootba/solver/linearizor_base.cpp"
    "${HGBP_ROOTBA_DIR}/src/rootba/solver/linearizor_power_sc.cpp"
    "${HGBP_ROOTBA_DIR}/src/rootba/solver/linearizor_sc.cpp"
    "${HGBP_ROOTBA_DIR}/src/rootba/cli/bal_cli_utils.cpp"
    "${HGBP_ROOTBA_DIR}/src/rootba/cli/cli_options.cpp")
  target_include_directories(powerba_capture PRIVATE "${HGBP_ROOTBA_PREFIX}/include" "${CMAKE_CURRENT_SOURCE_DIR}" "${HGBP_PROJECT}/benchmarks/baselines/ba/src" "${CMAKE_CURRENT_BINARY_DIR}/include"
    "${HGBP_ROOTBA_DIR}/external/clipp/clipp/include" "${HGBP_ABSL_PREFIX}/include")
  target_link_libraries(powerba_capture PRIVATE ${_rootba_runtime_libs} hgbp_ba_dependencies ${_absl_libs})
  target_compile_definitions(powerba_capture PRIVATE NDEBUG EIGEN_NO_DEBUG)
  if(MSVC)
    # Match the upstream RootBA core's Eigen alignment ABI, not the H-GBP AVX2 ABI.
    target_compile_options(powerba_capture PRIVATE /O2 /bigobj /Zc:__cplusplus /utf-8)
  else()
    target_compile_options(powerba_capture PRIVATE -O3)
  endif()
endif()
