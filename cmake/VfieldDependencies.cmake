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
    # Debian's netCDF package defines the imported target with a lowercase
    # spelling; accept both. Whichever target is used, its interface include
    # dirs resolve to the system prefix (/usr/include), which CMake turns
    # into -isystem /usr/include and breaks libstdc++'s #include_next in
    # every consuming TU. The system include path is a compiler default
    # anyway, so filter it out of the imported target.
    set(netcdf_target "")
    if(TARGET netCDF::netCDF)
      set(netcdf_target netCDF::netCDF)
    elseif(TARGET netCDF::netcdf)
      set(netcdf_target netCDF::netcdf)
    elseif(TARGET PkgConfig::netcdf)
      set(netcdf_target PkgConfig::netcdf)
    endif()
    if(netcdf_target)
      get_target_property(netcdf_iface_incs ${netcdf_target}
                          INTERFACE_INCLUDE_DIRECTORIES)
      set(netcdf_filtered_incs "")
      foreach(inc IN LISTS netcdf_iface_incs)
        if(NOT inc STREQUAL "/usr/include")
          list(APPEND netcdf_filtered_incs ${inc})
        endif()
      endforeach()
      set_property(TARGET ${netcdf_target} PROPERTY
                   INTERFACE_INCLUDE_DIRECTORIES "${netcdf_filtered_incs}")
      target_link_libraries(${target} PUBLIC ${netcdf_target})
    else()
      target_link_libraries(${target} PUBLIC ${netCDF_LIBRARIES})
      target_include_directories(${target} PUBLIC ${netCDF_INCLUDE_DIRS})
    endif()
  endif()
endfunction()
