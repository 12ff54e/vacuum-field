# VfieldDependencies.cmake — required and optional third-party dependencies.
#
# Defines:
#   CUDAToolkit          (required; imported target CUDA::cudart)
#   VFIELD_HAVE_NETCDF   (bool) — mgrid file reading compiled in
#   netCDF::netCDF / PkgConfig::netcdf / netCDF_*     (NetCDF link target)
#
# The optional backend is host-only: mgrid_provider.cpp is compiled by the C++
# compiler and linked only against the host library, never the CUDA targets.

find_package(CUDAToolkit 11 REQUIRED)

# ---- NetCDF ---------------------------------------------------------------
# Debian's libnetcdf-dev ships neither FindnetCDF.cmake nor a netCDFConfig
# package, so probe for a package first and fall back to pkg-config (the
# vmecpp pattern). find_package and pkg_check_modules share the netCDF_*
# variable namespace, so the fallback cleanly overwrites the failed probe.
set(VFIELD_HAVE_NETCDF FALSE)
if(VFIELD_USE_NETCDF)
  find_package(netCDF QUIET)
  if(NOT netCDF_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(netCDF QUIET IMPORTED_TARGET netcdf)
    endif()
  endif()
  if(netCDF_FOUND)
    set(VFIELD_HAVE_NETCDF TRUE)
    message(STATUS "vacuum-field: NetCDF mgrid reading enabled (netCDF ${netCDF_VERSION})")
  else()
    message(WARNING "vacuum-field: NetCDF not found - mgrid file reading disabled "
                    "(install libnetcdf-dev or pass -DVFIELD_USE_NETCDF=OFF)")
  endif()
endif()

# ---- helper: attach the optional NetCDF backend to a target ----------------
# Adds the backend sources and their library/defines to `target`. The backend
# TUs are guarded by VFIELD_HAVE_NETCDF, so the defines must be PUBLIC (the
# consumers of mgrid_provider.hpp read them too).
function(vfield_link_netcdf target)
  if(VFIELD_HAVE_NETCDF AND EXISTS
     ${PROJECT_SOURCE_DIR}/src/vfield/mgrid_provider.cpp)
    target_sources(${target} PRIVATE
      ${PROJECT_SOURCE_DIR}/src/vfield/mgrid_provider.cpp)
    target_compile_definitions(${target} PUBLIC VFIELD_HAVE_NETCDF)
    if(TARGET netCDF::netCDF)
      target_link_libraries(${target} PUBLIC netCDF::netCDF)
    elseif(TARGET PkgConfig::netcdf)
      target_link_libraries(${target} PUBLIC PkgConfig::netcdf)
    else()
      target_link_libraries(${target} PUBLIC ${netCDF_LIBRARIES})
      target_include_directories(${target} PUBLIC ${netCDF_INCLUDE_DIRS})
    endif()
  endif()
endfunction()
