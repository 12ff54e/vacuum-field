// cuda_status.hpp — centralized CUDA status handling.
//
// One throwing boundary for the whole runtime: `check_cuda` converts a
// non-success status into a `VfieldError` (a std::runtime_error the
// application boundary catches). Host-only (CUDA runtime API, no device code).
#ifndef VFIELD_RUNTIME_CUDA_STATUS_HPP_
#define VFIELD_RUNTIME_CUDA_STATUS_HPP_

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <string_view>

namespace vfield {

class VfieldError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

inline std::string cuda_error_string(cudaError_t err) {
    return std::string(cudaGetErrorString(err));
}

// Throws VfieldError when `err != cudaSuccess`; returns otherwise.
inline void check_cuda(cudaError_t err, std::string_view tag) {
    if (err != cudaSuccess) {
        throw VfieldError(std::string(tag) +
                          ": CUDA error: " + cuda_error_string(err));
    }
}

}  // namespace vfield

#endif  // VFIELD_RUNTIME_CUDA_STATUS_HPP_
