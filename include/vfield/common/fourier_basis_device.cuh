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
    using val_type = T;
    // Uploads the whole host basis. Called once at solver construction. The
    // host tables are double; convert to T explicitly — a raw cudaMemcpy
    // would reinterpret the doubles in float builds. For lasym the poloidal
    // tables are extended to the full [0, nThetaEven) range with the mirror
    // parity (cos(m*(2pi-theta)) = cos(m*theta), sin -> -sin) so the
    // full-range kernels can index them directly.
    explicit FourierBasisDevice(const FourierBasis& fb,
                                bool lasym,
                                int n_theta_even) {
        upload_table(fb.mscale, &mscale_);
        upload_table(fb.nscale, &nscale_);
        upload_table(fb.cosnv, &cosnv_);
        upload_table(fb.sinnv, &sinnv_);
        upload_table(fb.cosnvn, &cosnvn_);
        upload_table(fb.sinnvn, &sinnvn_);

        const int n_reduced =
            static_cast<int>(fb.cosmu.size()) / (fb.sizes().mnyq2 + 1);
        const int n_rows = lasym ? n_theta_even : n_reduced;
        upload_table_extended(fb.cosmu, n_reduced, n_rows, fb.sizes().mnyq2 + 1,
                            /*odd=*/false, &cosmu_);
        upload_table_extended(fb.sinmu, n_reduced, n_rows, fb.sizes().mnyq2 + 1,
                            /*odd=*/true, &sinmu_);
        upload_table_extended(fb.cosmum, n_reduced, n_rows, fb.sizes().mnyq2 + 1,
                            /*odd=*/true, &cosmum_);
        upload_table_extended(fb.sinmum, n_reduced, n_rows, fb.sizes().mnyq2 + 1,
                            /*odd=*/false, &sinmum_);
        upload_table_extended(fb.cosmui, n_reduced, n_rows, fb.sizes().mnyq2 + 1,
                            /*odd=*/false, &cosmui_);
        upload_table_extended(fb.sinmui, n_reduced, n_rows, fb.sizes().mnyq2 + 1,
                            /*odd=*/true, &sinmui_);
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
    static void upload_table(const std::vector<double>& src,
                            DeviceBuffer<T>* dst) {
        std::vector<T> converted(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) {
            converted[i] = static_cast<T>(src[i]);
        }
        dst->allocate(src.size());
        dst->upload(converted.data(), converted.size());
    }

    // Uploads a poloidal table with an optional mirror extension into the
    // second poloidal half [0, 2*pi[ (rows beyond n_reduced are the
    // parity-signed copies of the reflected reduced rows; the self-reflecting
    // endpoint rows are copied as-is, which the series parity guarantees is
    // correct).
    static void upload_table_extended(const std::vector<double>& src,
                                    int n_reduced,
                                    int n_rows,
                                    int num_m,
                                    bool odd,
                                    DeviceBuffer<T>* dst) {
        std::vector<T> extended(n_rows * num_m);
        for (int l = 0; l < n_rows; ++l) {
            int l_src = l;
            T sign = 1;
            if (l >= n_reduced) {
                l_src = (n_rows - l) % n_rows;
                sign = odd ? T(-1) : T(1);
            }
            for (int m = 0; m < num_m; ++m) {
                extended[l * num_m + m] =
                    sign * static_cast<T>(src[l_src * num_m + m]);
            }
        }
        dst->allocate(extended.size());
        dst->upload(extended.data(), extended.size());
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
