// fourier_basis_device.cuh — device-resident copies of the basis tables.
//
// Host-side (CUDA runtime API only, no kernels) holder that uploads the
// FourierBasis tables once at construction; the operator kernels read them
// read-only via data(). Header-only so both scalar instantiations share the
// code and no explicit-instantiation TU is needed.
#ifndef VFIELD_COMMON_FOURIER_BASIS_DEVICE_CUH_
#define VFIELD_COMMON_FOURIER_BASIS_DEVICE_CUH_

#include "vfield/common/fourier_basis.hpp"
#include "vfield/runtime/device_buffer.cuh"

#include <cuda_runtime.h>

namespace vfield {

template <class T>
class FourierBasisDevice {
   public:
    // Uploads the whole host basis. Called once at solver construction. The
    // host tables are double; convert to T explicitly — a raw cudaMemcpy
    // would reinterpret the doubles in float builds.
    explicit FourierBasisDevice(const FourierBasis& fb) {
        uploadTable(fb.mscale, &mscale_);
        uploadTable(fb.nscale, &nscale_);
        uploadTable(fb.cosmu, &cosmu_);
        uploadTable(fb.sinmu, &sinmu_);
        uploadTable(fb.cosmum, &cosmum_);
        uploadTable(fb.sinmum, &sinmum_);
        uploadTable(fb.cosmui, &cosmui_);
        uploadTable(fb.sinmui, &sinmui_);
        uploadTable(fb.cosnv, &cosnv_);
        uploadTable(fb.sinnv, &sinnv_);
        uploadTable(fb.cosnvn, &cosnvn_);
        uploadTable(fb.sinnvn, &sinnvn_);
    }

    const T* mscale() const { return mscale_.data(); }
    const T* nscale() const { return nscale_.data(); }
    const T* cosmu() const { return cosmu_.data(); }
    const T* sinmu() const { return sinmu_.data(); }
    const T* cosmum() const { return cosmum_.data(); }
    const T* sinmum() const { return sinmum_.data(); }
    const T* cosmui() const { return cosmui_.data(); }
    const T* sinmui() const { return sinmui_.data(); }
    const T* cosnv() const { return cosnv_.data(); }
    const T* sinnv() const { return sinnv_.data(); }
    const T* cosnvn() const { return cosnvn_.data(); }
    const T* sinnvn() const { return sinnvn_.data(); }

   private:
    static void uploadTable(const std::vector<double>& src,
                            DeviceBuffer<T>* dst) {
        std::vector<T> converted(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) {
            converted[i] = static_cast<T>(src[i]);
        }
        dst->allocate(src.size());
        dst->upload(converted.data(), converted.size());
    }

    DeviceBuffer<T> mscale_;
    DeviceBuffer<T> nscale_;
    DeviceBuffer<T> cosmu_;
    DeviceBuffer<T> sinmu_;
    DeviceBuffer<T> cosmum_;
    DeviceBuffer<T> sinmum_;
    DeviceBuffer<T> cosmui_;
    DeviceBuffer<T> sinmui_;
    DeviceBuffer<T> cosnv_;
    DeviceBuffer<T> sinnv_;
    DeviceBuffer<T> cosnvn_;
    DeviceBuffer<T> sinnvn_;
};

}  // namespace vfield

#endif  // VFIELD_COMMON_FOURIER_BASIS_DEVICE_CUH_
