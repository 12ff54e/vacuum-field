# tests/CMakeLists.txt — standalone test executables, no framework.
#
# Host-only tests (no CUDA includes) link vfield_host alone and carry the
# `host` label so the CPU CI can run them without a GPU. Device tests link
# both scalar libraries and carry `golden` where they read tests/data/.

add_executable(test_sizes test_sizes.cpp)
target_include_directories(test_sizes PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include)
target_link_libraries(test_sizes PRIVATE vfield_host)
vfield_enable_warnings(test_sizes)
vfield_add_test(test_sizes host)

add_executable(test_fourier_basis test_fourier_basis.cpp)
target_include_directories(test_fourier_basis PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include)
target_link_libraries(test_fourier_basis PRIVATE vfield_host)
vfield_enable_warnings(test_fourier_basis)
vfield_add_test(test_fourier_basis host)

add_executable(test_singular_coefficients test_singular_coefficients.cpp)
target_include_directories(test_singular_coefficients PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include)
target_link_libraries(test_singular_coefficients PRIVATE vfield_host)
vfield_enable_warnings(test_singular_coefficients)
vfield_add_test(test_singular_coefficients host)

# Device tests: link both scalar libraries; the support headers live in
# tests/support.
add_executable(test_surface_geometry test_surface_geometry.cu)
target_include_directories(test_surface_geometry PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include
  ${PROJECT_SOURCE_DIR}/tests/support)
target_link_libraries(test_surface_geometry PRIVATE
  vfield_host vfield_cuda_double vfield_cuda_float)
vfield_enable_warnings(test_surface_geometry)
vfield_add_test(test_surface_geometry unit)

add_executable(test_surface_geometry_golden test_surface_geometry_golden.cu)
target_include_directories(test_surface_geometry_golden PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include
  ${PROJECT_SOURCE_DIR}/tests/support)
target_link_libraries(test_surface_geometry_golden PRIVATE
  vfield_host vfield_cuda_double)
vfield_enable_warnings(test_surface_geometry_golden)
vfield_add_test(test_surface_geometry_golden unit;golden)

if(VFIELD_HAVE_NETCDF)
  add_executable(test_external_field_golden test_external_field_golden.cu)
  target_include_directories(test_external_field_golden PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include
    ${PROJECT_SOURCE_DIR}/tests/support)
  target_link_libraries(test_external_field_golden PRIVATE
    vfield_host vfield_cuda_double)
  vfield_enable_warnings(test_external_field_golden)
  vfield_add_test(test_external_field_golden unit;golden)
endif()

add_executable(test_singular_integrals test_singular_integrals.cu)
target_include_directories(test_singular_integrals PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include
  ${PROJECT_SOURCE_DIR}/tests/support)
target_link_libraries(test_singular_integrals PRIVATE
  vfield_host vfield_cuda_double vfield_cuda_float)
vfield_enable_warnings(test_singular_integrals)
vfield_add_test(test_singular_integrals unit)

add_executable(test_singular_golden test_singular_golden.cu)
target_include_directories(test_singular_golden PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include
  ${PROJECT_SOURCE_DIR}/tests/support)
target_link_libraries(test_singular_golden PRIVATE
  vfield_host vfield_cuda_double)
vfield_enable_warnings(test_singular_golden)
vfield_add_test(test_singular_golden unit;golden)

add_executable(test_regularized_axisym test_regularized_axisym.cu)
target_include_directories(test_regularized_axisym PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include
  ${PROJECT_SOURCE_DIR}/tests/support)
target_link_libraries(test_regularized_axisym PRIVATE
  vfield_host vfield_cuda_double vfield_cuda_float)
vfield_enable_warnings(test_regularized_axisym)
vfield_add_test(test_regularized_axisym unit)

add_executable(test_regularized_golden test_regularized_golden.cu)
target_include_directories(test_regularized_golden PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include
  ${PROJECT_SOURCE_DIR}/tests/support)
target_link_libraries(test_regularized_golden PRIVATE
  vfield_host vfield_cuda_double)
vfield_enable_warnings(test_regularized_golden)
vfield_add_test(test_regularized_golden unit;golden)

add_executable(test_lu test_lu.cpp)
target_include_directories(test_lu PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include)
target_link_libraries(test_lu PRIVATE vfield_host)
vfield_enable_warnings(test_lu)
vfield_add_test(test_lu host)

add_executable(test_laplace_solver test_laplace_solver.cu)
target_include_directories(test_laplace_solver PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include
  ${PROJECT_SOURCE_DIR}/tests/support)
target_link_libraries(test_laplace_solver PRIVATE
  vfield_host vfield_cuda_double vfield_cuda_float)
vfield_enable_warnings(test_laplace_solver)
vfield_add_test(test_laplace_solver unit)

add_executable(test_laplace_golden test_laplace_golden.cu)
target_include_directories(test_laplace_golden PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include
  ${PROJECT_SOURCE_DIR}/tests/support)
target_link_libraries(test_laplace_golden PRIVATE
  vfield_host vfield_cuda_double)
vfield_enable_warnings(test_laplace_golden)
vfield_add_test(test_laplace_golden unit;golden)

add_executable(test_lasym_parity test_lasym_parity.cu)
target_include_directories(test_lasym_parity PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include
  ${PROJECT_SOURCE_DIR}/tests/support)
target_link_libraries(test_lasym_parity PRIVATE
  vfield_host vfield_cuda_double)
vfield_enable_warnings(test_lasym_parity)
vfield_add_test(test_lasym_parity unit)

if(VFIELD_HAVE_NETCDF)
  add_executable(test_vacuum_field_golden test_vacuum_field_golden.cu)
  target_include_directories(test_vacuum_field_golden PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include
    ${PROJECT_SOURCE_DIR}/tests/support)
  target_link_libraries(test_vacuum_field_golden PRIVATE
    vfield_host vfield_cuda_double)
  vfield_enable_warnings(test_vacuum_field_golden)
  vfield_add_test(test_vacuum_field_golden unit;golden)

  add_executable(test_float_smoke test_float_smoke.cu)
  target_include_directories(test_float_smoke PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include
    ${PROJECT_SOURCE_DIR}/tests/support)
  target_link_libraries(test_float_smoke PRIVATE
    vfield_host vfield_cuda_double vfield_cuda_float)
  vfield_enable_warnings(test_float_smoke)
  vfield_add_test(test_float_smoke unit)
endif()
