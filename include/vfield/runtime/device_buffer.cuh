// device_buffer.cuh — movable, non-copyable RAII device allocation.
//
// Owns a cudaMalloc'd span of `T`. Move-only: moving transfers the raw pointer
// and nulls the source, so assigning a freshly-allocated buffer over a live one
// frees the old allocation exactly once. No allocation happens in the hot
// loop: every buffer is constructed once at solver setup and its `data()` is
// passed to the kernels unchanged. A trimmed port of cuMES's runtime pattern
// (no arena, no pinned buffers — the library's device footprint is a few MB).
#ifndef VFIELD_RUNTIME_DEVICE_BUFFER_CUH_
#define VFIELD_RUNTIME_DEVICE_BUFFER_CUH_

#include "vfield/runtime/cuda_status.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace vfield {

template <class T>
class DeviceBuffer {
   public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) { allocate(count); }

    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), count_(other.count_), owned_(other.owned_) {
        other.data_ = nullptr;
        other.count_ = 0;
        other.owned_ = true;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            count_ = other.count_;
            owned_ = other.owned_;
            other.data_ = nullptr;
            other.count_ = 0;
            other.owned_ = true;
        }
        return *this;
    }

    void allocate(std::size_t count) {
        release();
        if (count == 0) return;
        check_cuda(cudaMalloc(&data_, count * sizeof(T)),
                   "DeviceBuffer::allocate");
        count_ = count;
        owned_ = true;
    }

    void release() {
        // A free failure is not recoverable and must not throw from a
        // destructor path; the allocation/transfer/copy entry points are the
        // checked boundary.
        if (data_ != nullptr && owned_) { cudaFree(data_); }
        data_ = nullptr;
        count_ = 0;
        owned_ = true;
    }

    void zero() {
        if (count_ != 0) {
            check_cuda(cudaMemset(data_, 0, count_ * sizeof(T)),
                       "DeviceBuffer::zero");
        }
    }

    // Host-to-device upload of a contiguous host range.
    template <class U>
    void upload(const U* src, std::size_t count) {
        if (count != count_) {
            throw VfieldError("DeviceBuffer::upload: size mismatch");
        }
        if (count_ == 0) return;
        check_cuda(
            cudaMemcpy(data_, src, count_ * sizeof(T), cudaMemcpyHostToDevice),
            "DeviceBuffer::upload");
    }

    // Device-to-host download into a contiguous host range.
    template <class U>
    void download(U* dst, std::size_t count) const {
        if (count != count_) {
            throw VfieldError("DeviceBuffer::download: size mismatch");
        }
        if (count_ == 0) return;
        check_cuda(
            cudaMemcpy(dst, data_, count_ * sizeof(T), cudaMemcpyDeviceToHost),
            "DeviceBuffer::download");
    }

    // One device-to-device copy; requires identical element counts.
    void copy_from(const DeviceBuffer& other) {
        if (other.count_ != count_) {
            throw VfieldError("DeviceBuffer::copy_from: size mismatch");
        }
        if (count_ == 0) return;
        check_cuda(cudaMemcpy(data_, other.data_, count_ * sizeof(T),
                              cudaMemcpyDeviceToDevice),
                   "DeviceBuffer::copy_from");
    }

    T* data() const { return data_; }
    std::size_t size() const { return count_; }
    std::size_t byte_size() const { return count_ * sizeof(T); }
    bool empty() const { return count_ == 0; }

   private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
    bool owned_ = true;
};

}  // namespace vfield

#endif  // VFIELD_RUNTIME_DEVICE_BUFFER_CUH_
