# VfieldOptions.cmake — vacuum-field user-facing build options.
#
# Every option is a cache variable so CMakePresets.json / -D<VAR> on the
# command line can set them. An embedding consumer (cuMES) propagates its own
# scalar switch into VFIELD_USE_FLOAT before add_subdirectory.

# Precision: all computation is templated on T; `Real` (vfield_types.h) is
# the compile-time switch. OFF = double (verified default), ON = float
# (experimental). The host dense solve stays double in both builds.
option(VFIELD_USE_FLOAT "Build the float instantiation as the selected library (Real)" OFF)

# Optional NetCDF mgrid reading. Compiled in only if the option is ON and the
# library is found (see VfieldDependencies.cmake); LoadFile reports an error
# when NetCDF support is not compiled in.
option(VFIELD_USE_NETCDF "Enable NetCDF mgrid file reading" ON)

# Test suite (standalone executables, no framework). An embedding consumer
# can turn this off to skip the library's own tests in its ctest run.
option(VFIELD_BUILD_TESTS "Build the vacuum-field test suite" ON)

# Promote warnings to errors for library sources.
option(VFIELD_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
