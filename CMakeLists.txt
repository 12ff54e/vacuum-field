cmake_minimum_required(VERSION 3.20)

# ---- cmake modules ---------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Use g++-12 as CUDA host compiler when the toolchain does not pin one (CUDA
# 12.1 does not support gcc > 12). Defaulted only-if-unset: an embedding
# consumer (cuMES) sets its own pin before project() and this guard honours
# it. Only applied when g++-12 actually exists.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  if(EXISTS /usr/bin/g++-12)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12 CACHE FILEPATH
        "CUDA host compiler")
  endif()
endif()

# CUDA architectures must be defaulted BEFORE project()/enable_language(CUDA)
# (CMake otherwise substitutes its own sm_52 fallback). The default applies
# only when the consumer has not already provided CMAKE_CUDA_ARCHITECTURES —
# cuMES, which embeds this library via add_subdirectory, sets its own.
include(VfieldCudaArchitectures)

project(vacuum-field VERSION 0.1.0 LANGUAGES CXX CUDA)

include(VfieldOptions)
include(VfieldDependencies)
include(VfieldWarnings)
include(VfieldTest)

# ---- language standards -----------------------------------------------------
# C++20 for host C++ and CUDA TUs alike; GNU extensions off (strict -std=c++20).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ---- host-only library ------------------------------------------------------
# Sizes, FourierBasis, singular-kernel coefficients, mgrid reading, and the
# host dense solve. Plain C++ — compiled by the host compiler, no device code.
add_library(vfield_host STATIC
  src/vfield/sizes.cpp
  src/vfield/fourier_basis.cpp
  src/vfield/singular_coefficients.cpp
  src/vfield/lu_solve.cpp
  src/vfield/json_parser.cpp)
target_include_directories(vfield_host PUBLIC ${PROJECT_SOURCE_DIR}/include)
vfield_enable_warnings(vfield_host)
vfield_link_netcdf(vfield_host)

# ---- CUDA operator libraries (one scalar type each) -------------------------
# Templated operator modules split into a *_double.cu and a *_float.cu
# explicit-instantiation TU (one scalar type per TU). The consumer links only
# the TU matching `Real`; tests link both (they instantiate both).
set(VFIELD_CUDA_MODULES
  surface_geometry singular_integrals regularized_integrals laplace_solver
  external_field vacuum_field_solver)
set(VFIELD_CUDA_DOUBLE_SOURCES "")
set(VFIELD_CUDA_FLOAT_SOURCES "")
foreach(mod IN LISTS VFIELD_CUDA_MODULES)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/${mod}_double.cu)
    list(APPEND VFIELD_CUDA_DOUBLE_SOURCES src/${mod}_double.cu)
    list(APPEND VFIELD_CUDA_FLOAT_SOURCES src/${mod}_float.cu)
  endif()
endforeach()

if(VFIELD_CUDA_DOUBLE_SOURCES)
  add_library(vfield_cuda_double STATIC ${VFIELD_CUDA_DOUBLE_SOURCES})
  target_include_directories(vfield_cuda_double PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(vfield_cuda_double PUBLIC CUDA::cudart vfield_host)
  vfield_enable_warnings(vfield_cuda_double)

  add_library(vfield_cuda_float STATIC ${VFIELD_CUDA_FLOAT_SOURCES})
  target_include_directories(vfield_cuda_float PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(vfield_cuda_float PUBLIC CUDA::cudart vfield_host)
  vfield_enable_warnings(vfield_cuda_float)
endif()

if(VFIELD_CUDA_DOUBLE_SOURCES)
  if(VFIELD_USE_FLOAT)
    set(VFIELD_SELECTED_CUDA vfield_cuda_float)
  else()
    set(VFIELD_SELECTED_CUDA vfield_cuda_double)
  endif()

  # The consumer-facing alias: vfield::vfield resolves to the scalar library
  # matching the VFIELD_USE_FLOAT switch.
  add_library(vfield::vfield ALIAS ${VFIELD_SELECTED_CUDA})
endif()

# ---- tests ------------------------------------------------------------------
if(VFIELD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
