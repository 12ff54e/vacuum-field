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
    // Uploads the whole host basis. Called once at solver construction.
    explicit FourierBasisDevice(const FourierBasis& fb) {
        mscale_.allocate(fb.mscale.size());
        nscale_.allocate(fb.nscale.size());
        cosmu_.allocate(fb.cosmu.size());
        sinmu_.allocate(fb.sinmu.size());
        cosmum_.allocate(fb.cosmum.size());
        sinmum_.allocate(fb.sinmum.size());
        cosmui_.allocate(fb.cosmui.size());
        sinmui_.allocate(fb.sinmui.size());
        cosnv_.allocate(fb.cosnv.size());
        sinnv_.allocate(fb.sinnv.size());
        cosnvn_.allocate(fb.cosnvn.size());
        sinnvn_.allocate(fb.sinnvn.size());

        mscale_.upload(fb.mscale.data(), fb.mscale.size());
        nscale_.upload(fb.nscale.data(), fb.nscale.size());
        cosmu_.upload(fb.cosmu.data(), fb.cosmu.size());
        sinmu_.upload(fb.sinmu.data(), fb.sinmu.size());
        cosmum_.upload(fb.cosmum.data(), fb.cosmum.size());
        sinmum_.upload(fb.sinmum.data(), fb.sinmum.size());
        cosmui_.upload(fb.cosmui.data(), fb.cosmui.size());
        sinmui_.upload(fb.sinmui.data(), fb.sinmui.size());
        cosnv_.upload(fb.cosnv.data(), fb.cosnv.size());
        sinnv_.upload(fb.sinnv.data(), fb.sinnv.size());
        cosnvn_.upload(fb.cosnvn.data(), fb.cosnvn.size());
        sinnvn_.upload(fb.sinnvn.data(), fb.sinnvn.size());
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
