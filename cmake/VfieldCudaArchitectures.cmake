# VfieldCudaArchitectures.cmake — CUDA architecture selection.
#
# Defaults to the Pascal/Turing/Ampere/Ada set (61/75/80/86/89), trimmed to
# what the configured toolkit can actually compile: sm_86 requires CUDA >=
# 11.1, sm_89 requires CUDA >= 11.8. This module runs BEFORE
# project()/enable_language(CUDA) because CMake's CUDA support populates
# CMAKE_CUDA_ARCHITECTURES with its own fallback (sm_52) during
# enable_language if it is unset, and a post-project non-FORCE cache set
# cannot override that already-cached value. The guard honours an explicit
# -DCMAKE_CUDA_ARCHITECTURES (or the environment variable), so an embedding
# consumer (cuMES) keeps control of the architecture set.

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CMAKE_CUDA_ARCHITECTURES})
  # Base set: every architecture supported by the CUDA 11.0 floor.
  set(VFIELD_DEFAULT_ARCHS "61;75;80")
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.1")
      list(APPEND VFIELD_DEFAULT_ARCHS 86)
    endif()
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
      list(APPEND VFIELD_DEFAULT_ARCHS 89)
    endif()
  else()
    # The toolkit is not locatable yet (VfieldDependencies.cmake's REQUIRED
    # find_package runs after project()). Fall back to the conservative
    # CUDA 11.0-compatible set and say so.
    message(STATUS "vacuum-field: CUDA toolkit not yet locatable; defaulting "
                   "CMAKE_CUDA_ARCHITECTURES to the CUDA 11.0-compatible set "
                   "${VFIELD_DEFAULT_ARCHS}. Pass -DCMAKE_CUDA_ARCHITECTURES="
                   "61;75;80;86;89 to include Ampere/Ada on a newer toolkit.")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "${VFIELD_DEFAULT_ARCHS}" CACHE STRING
      "CUDA architectures to build for (semicolon-separated compute capabilities)")
endif()
set_property(CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY STRINGS
    "61;75;80;86;89" "61" "75" "80" "86" "89")
