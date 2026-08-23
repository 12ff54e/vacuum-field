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
