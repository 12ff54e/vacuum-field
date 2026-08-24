// surface_geometry_operator.hpp — LCFS real-space geometry on the device.
//
// Port of vmecpp's SurfaceGeometry (free_boundary/surface_geometry): inverse
// Fourier synthesis of the LCFS coefficients onto the surface grid plus the
// derived normal / metric / second-fundamental-form quantities, with the
// same normalization conventions (guv = 2*g_theta_phi/nfp, gvv =
// g_phi_phi/nfp^2, sn = sign_of_jacobian * (-r*N)). Serial over the whole
// grid: the OpenMP tangential partitioning is dropped, so every array is
// indexed by the flat zeta-fast index kl = l*nZeta + k over the effective
// poloidal range [0, nThetaEff).
#ifndef VFIELD_FREE_BOUNDARY_SURFACE_GEOMETRY_OPERATOR_HPP_
#define VFIELD_FREE_BOUNDARY_SURFACE_GEOMETRY_OPERATOR_HPP_

#include "vfield/common/fourier_basis_device.cuh"
#include "vfield/common/sizes.hpp"
#include "vfield/runtime/device_buffer.cuh"

namespace vfield {

template <class T>
class SurfaceGeometryOperator {
   public:
    using val_type = T;
    SurfaceGeometryOperator(const Sizes& sizes,
                            const FourierBasisDevice<T>& fb);

    // Evaluates the Fourier series for the surface geometry and computes the
    // derived quantities. The eight coefficient arrays are device pointers to
    // mnsize-sized n-major arrays (index n*mpol+m); the antisymmetric arrays
    // may be nullptr when !lasym. full_update also computes the second
    // derivatives and the rzb2/rcosuv/rsinuv arrays.
    void update(const T* d_rcc,
                const T* d_rss,
                const T* d_rsc,
                const T* d_rcs,
                const T* d_zsc,
                const T* d_zcs,
                const T* d_zcc,
                const T* d_zss,
                int sign_of_jacobian,
                bool full_update);

    // [nThetaEven * nZeta] — full surface (the symmetric half is mirrored).
    const T* r1b() const { return r1b_.data(); }
    const T* z1b() const { return z1b_.data(); }
    const T* rcosuv() const { return rcosuv_.data(); }
    const T* rsinuv() const { return rsinuv_.data(); }
    const T* rzb2() const { return rzb2_.data(); }

    // [nZnT] — effective poloidal range.
    const T* rub() const { return rub_.data(); }
    const T* rvb() const { return rvb_.data(); }
    const T* zub() const { return zub_.data(); }
    const T* zvb() const { return zvb_.data(); }
    const T* ruu() const { return ruu_.data(); }
    const T* ruv() const { return ruv_.data(); }
    const T* rvv() const { return rvv_.data(); }
    const T* zuu() const { return zuu_.data(); }
    const T* zuv() const { return zuv_.data(); }
    const T* zvv() const { return zvv_.data(); }

    // [nfp] / [nZeta] — field-period and grid toroidal-angle trig tables.
    const T* cos_per() const { return cos_per_.data(); }
    const T* sin_per() const { return sin_per_.data(); }
    const T* cos_phi() const { return cos_phi_.data(); }
    const T* sin_phi() const { return sin_phi_.data(); }

    const T* snr() const { return snr_.data(); }
    const T* snv() const { return snv_.data(); }
    const T* snz() const { return snz_.data(); }
    const T* guu() const { return guu_.data(); }
    const T* guv() const { return guv_.data(); }
    const T* gvv() const { return gvv_.data(); }
    const T* auu() const { return auu_.data(); }
    const T* auv() const { return auv_.data(); }
    const T* avv() const { return avv_.data(); }
    const T* drv() const { return drv_.data(); }

   private:
    Sizes sizes_;
    const FourierBasisDevice<T>& fb_;
    // All launches go on this stream (default stream when nullptr).
    cudaStream_t stream_;

    // [nfp] / [nZeta] — field-period and grid toroidal-angle trig tables.
    DeviceBuffer<T> cos_per_;
    DeviceBuffer<T> sin_per_;
    DeviceBuffer<T> cos_phi_;
    DeviceBuffer<T> sin_phi_;

    // Full surface [nThetaEven * nZeta].
    DeviceBuffer<T> r1b_;
    DeviceBuffer<T> z1b_;
    DeviceBuffer<T> rcosuv_;
    DeviceBuffer<T> rsinuv_;
    DeviceBuffer<T> rzb2_;

    // Effective poloidal range [nZnT].
    DeviceBuffer<T> rub_;
    DeviceBuffer<T> rvb_;
    DeviceBuffer<T> zub_;
    DeviceBuffer<T> zvb_;
    DeviceBuffer<T> ruu_;
    DeviceBuffer<T> ruv_;
    DeviceBuffer<T> rvv_;
    DeviceBuffer<T> zuu_;
    DeviceBuffer<T> zuv_;
    DeviceBuffer<T> zvv_;
    DeviceBuffer<T> snr_;
    DeviceBuffer<T> snv_;
    DeviceBuffer<T> snz_;
    DeviceBuffer<T> guu_;
    DeviceBuffer<T> guv_;
    DeviceBuffer<T> gvv_;
    DeviceBuffer<T> auu_;
    DeviceBuffer<T> auv_;
    DeviceBuffer<T> avv_;
    DeviceBuffer<T> drv_;

    // lasym-only antisymmetric pieces [nZnT] (only the reduced poloidal range
    // is written; the mirror kernels read them there).
    DeviceBuffer<T> r1b_asym_;
    DeviceBuffer<T> z1b_asym_;
    DeviceBuffer<T> rub_asym_;
    DeviceBuffer<T> rvb_asym_;
    DeviceBuffer<T> zub_asym_;
    DeviceBuffer<T> zvb_asym_;
    DeviceBuffer<T> ruu_asym_;
    DeviceBuffer<T> ruv_asym_;
    DeviceBuffer<T> rvv_asym_;
    DeviceBuffer<T> zuu_asym_;
    DeviceBuffer<T> zuv_asym_;
    DeviceBuffer<T> zvv_asym_;
};

}  // namespace vfield

#endif  // VFIELD_FREE_BOUNDARY_SURFACE_GEOMETRY_OPERATOR_HPP_
