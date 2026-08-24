// singular_integrals_operator.hpp — singular part of the Green's-function
// integrals on the device.
//
// Port of vmecpp's SingularIntegrals (free_boundary/singular_integrals): the
// analytically integrable part of the boundary-integral kernels. Per surface
// point, the one-dimensional integrals T_l^+/- follow a three-term recurrence
// from the local quadratic forms of the metric (forward, or Miller's
// backward recurrence when the forward branch is unstable), and S_l^+/- are
// the surface-weighted combinations with the second fundamental form. They
// feed the Laplace system as bvec (the singular part of the RHS, from
// b_dot_n) and grpmn (the singular part of the system-matrix kernel).
// nf = ntor, mf = mpol + 1 (the tangential Fourier resolution of NESTOR).
#ifndef VFIELD_FREE_BOUNDARY_SINGULAR_INTEGRALS_OPERATOR_HPP_
#define VFIELD_FREE_BOUNDARY_SINGULAR_INTEGRALS_OPERATOR_HPP_

#include "vfield/common/fourier_basis_device.cuh"
#include "vfield/common/singular_coefficients.hpp"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/surface_geometry_operator.hpp"
#include "vfield/runtime/device_buffer.cuh"

namespace vfield {

template <class T>
class SingularIntegralsOperator {
   public:
    using val_type = T;
    SingularIntegralsOperator(const Sizes& sizes,
                              const FourierBasisDevice<T>& fb,
                              const SurfaceGeometryOperator<T>& sg);

    // b_dot_n is a device pointer [nZnT] (the external-field normal component).
    // full_update also recomputes the second-fundamental-form coefficients
    // and the grpmn kernel part.
    void update(const T* d_bdotn, bool full_update);

    // Test/expert entry points mirroring vmecpp's prepare_update/perform_update
    // split: the local quadratic forms of the metric (a, b2, c) and the
    // second fundamental form (A, B2, C) are passed explicitly instead of
    // being read from the surface geometry (used by the synthetic-geometry
    // unit tests). update() delegates to both.
    void prepare_update(const T* d_a,
                       const T* d_b2,
                       const T* d_c,
                       const T* d_A,
                       const T* d_B2,
                       const T* d_C,
                       bool full_update);
    void perform_update(const T* d_bdotn, bool full_update);

    // [mnfull = (2nf+1)(mf+1)] — the singular part of the Laplace RHS.
    const T* bvec_sin() const { return bvec_sin_.data(); }
    const T* bvec_cos() const { return bvec_cos_.data(); }
    // [mnfull * nZnT] — the singular part of the system-matrix kernel.
    const T* grpmn_sin() const { return grpmn_sin_.data(); }
    const T* grpmn_cos() const { return grpmn_cos_.data(); }
    // [(nf+mf+1) * nZnT] — the recurrence integrals per point (golden check).
    const T* tlp() const { return tlp_.data(); }
    const T* tlm() const { return tlm_.data(); }
    const T* slp() const { return slp_.data(); }
    const T* slm() const { return slm_.data(); }

    int nf() const { return nf_; }
    int mf() const { return mf_; }

   private:
    Sizes sizes_;
    const FourierBasisDevice<T>& fb_;
    const SurfaceGeometryOperator<T>& sg_;
    cudaStream_t stream_;

    int nf_;
    int mf_;
    int mnfull_;

    // cmns coefficients + integration weights on device.
    DeviceBuffer<T> cmns_;
    DeviceBuffer<T> w_int_;

    // Per-point constants and T/S tables.
    DeviceBuffer<T> sqrtc2_;
    DeviceBuffer<T> sqrta2_;
    DeviceBuffer<T> r1p_;
    DeviceBuffer<T> r1m_;
    DeviceBuffer<T> r0p_;
    DeviceBuffer<T> r0m_;
    DeviceBuffer<T> ra1p_;
    DeviceBuffer<T> ra1m_;
    DeviceBuffer<T> tlp_;  // [(nf+mf+2) * nZnT] — one extra entry for the
    DeviceBuffer<T> tlm_;  // last iteration of the fl loop
    DeviceBuffer<T> slp_;  // [(nf+mf+1) * nZnT]
    DeviceBuffer<T> slm_;

    DeviceBuffer<T> bvec_sin_;
    DeviceBuffer<T> bvec_cos_;
    DeviceBuffer<T> grpmn_sin_;
    DeviceBuffer<T> grpmn_cos_;
};

}  // namespace vfield

#endif  // VFIELD_FREE_BOUNDARY_SINGULAR_INTEGRALS_OPERATOR_HPP_
