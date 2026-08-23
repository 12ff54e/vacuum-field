// vfield_test_cuda_helper.cuh — CUDA test support: staging and comparison.
//
// Includes vfield_test.h for the assertion harness and adds device staging
// helpers plus the relative-absolute closeness predicate (vmecpp's
// IsCloseRelAbs semantics) used by the golden comparisons.
#ifndef VFIELD_TESTS_SUPPORT_VFIELD_TEST_CUDA_HELPER_CUH_
#define VFIELD_TESTS_SUPPORT_VFIELD_TEST_CUDA_HELPER_CUH_

#include "vfield/runtime/cuda_status.hpp"
#include "vfield/runtime/device_buffer.cuh"
#include "vfield_test.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace vfield::test {

// Stages a host vector on the device (owning buffer).
template <class T>
DeviceBuffer<T> toDevice(const std::vector<T>& h) {
    DeviceBuffer<T> d(h.size());
    d.upload(h.data(), h.size());
    return d;
}

// Downloads a device buffer into a host vector.
template <class T>
std::vector<T> toHost(const DeviceBuffer<T>& d) {
    std::vector<T> h(d.size());
    d.download(h.data(), h.size());
    return h;
}

// Downloads a raw device pointer into a host vector.
template <class T>
std::vector<T> toHost(const T* d, std::size_t n) {
    std::vector<T> h(n);
    check_cuda(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost),
               "toHost");
    return h;
}

}  // namespace vfield::test

#endif  // VFIELD_TESTS_SUPPORT_VFIELD_TEST_CUDA_HELPER_CUH_
