cmake_minimum_required(VERSION 3.20)
project(flipster VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(FLIPSTER_ENABLE_CUDA "AUTO" CACHE STRING "Build the CUDA backend: AUTO, ON or OFF")
set_property(CACHE FLIPSTER_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)
option(FLIPSTER_BUILD_PYTHON "Build the pybind11 module" OFF)
option(FLIPSTER_BUILD_TESTS "Build the GoogleTest suite" ${PROJECT_IS_TOP_LEVEL})

# ---------------------------------------------------------------- CUDA detection
set(FLIPSTER_WITH_CUDA OFF)
if(NOT FLIPSTER_ENABLE_CUDA STREQUAL "OFF")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # Turing, Ampere, Ada. Each also embeds PTX, so newer GPUs JIT-compile.
      # Pass -DCMAKE_CUDA_ARCHITECTURES=native to target just your own card.
      set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(FLIPSTER_WITH_CUDA ON)
  elseif(FLIPSTER_ENABLE_CUDA STREQUAL "ON")
    message(FATAL_ERROR "FLIPSTER_ENABLE_CUDA=ON but no CUDA compiler (nvcc) was found")
  endif()
endif()
message(STATUS "flipster: CUDA backend ${FLIPSTER_WITH_CUDA}")

# ---------------------------------------------------------------- core library
find_package(OpenMP)

add_library(flipster_core STATIC
  core/src/engine.cpp
  core/src/cpu/engine_cpu.cpp)
target_include_directories(flipster_core
  PUBLIC core/include
  PRIVATE core/src)
if(OpenMP_CXX_FOUND)
  target_link_libraries(flipster_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(MSVC)
  target_compile_options(flipster_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2 /fp:precise>)
else()
  target_compile_options(flipster_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>)
endif()

if(FLIPSTER_WITH_CUDA)
  target_sources(flipster_core PRIVATE core/src/cuda/engine_cuda.cu)
  target_compile_definitions(flipster_core PUBLIC FLIPSTER_WITH_CUDA)
  set_target_properties(flipster_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  # --fmad=false keeps GPU arithmetic bit-comparable with the CPU engine.
  target_compile_options(flipster_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -lineinfo>)
  target_link_libraries(flipster_core PUBLIC CUDA::cudart)
endif()

# ---------------------------------------------------------------- python module
if(FLIPSTER_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core MODULE bindings/module.cpp)
  target_link_libraries(_core PRIVATE flipster_core)
  install(TARGETS _core LIBRARY DESTINATION flipster RUNTIME DESTINATION flipster)
endif()

# ---------------------------------------------------------------- tests / tools
if(FLIPSTER_BUILD_TESTS)
  find_package(GTest CONFIG QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  enable_testing()
  add_executable(flipster_tests
    core/tests/test_ops.cpp
    core/tests/test_flow.cpp
    core/tests/test_synthesis.cpp
    core/tests/test_cuda_parity.cpp)
  target_include_directories(flipster_tests PRIVATE core/src core/tests)
  target_link_libraries(flipster_tests PRIVATE flipster_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(flipster_tests DISCOVERY_TIMEOUT 60)

  add_executable(flipster_bench core/tools/bench.cpp)
  target_include_directories(flipster_bench PRIVATE core/tests)
  target_link_libraries(flipster_bench PRIVATE flipster_core)
endif()
