// regularized_integrals_operator.hpp — regularized Green's-function part on
// the device.
//
// Port of vmecpp's RegularizedIntegrals (free_boundary/regularized_
// integrals): the smooth difference kernel (exact Green's function minus an
// analytic local approximation in the tan-half-angle variables), evaluated
// over the whole surface and the nfp field-period images. greenp holds the
// regularized normal-derivative kernel per source point, gstore the
// regularized source term (b_dot_n-weighted). The exact singularity (source
// point, first period) is skipped — it is handled analytically in
// SingularIntegralsOperator. For axisymmetric plasmas (nZeta == 1) the
// toroidal integral is performed over nvper = 64 equally-spaced toroidal
// images instead.
#ifndef VFIELD_FREE_BOUNDARY_REGULARIZED_INTEGRALS_OPERATOR_HPP_
#define VFIELD_FREE_BOUNDARY_REGULARIZED_INTEGRALS_OPERATOR_HPP_

#include "vfield/common/fourier_basis_device.cuh"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/surface_geometry_operator.hpp"
#include "vfield/runtime/device_buffer.cuh"

namespace vfield {

template <class T>
class RegularizedIntegralsOperator {
   public:
    using val_type = T;
    RegularizedIntegralsOperator(const Sizes& sizes,
                                 const SurfaceGeometryOperator<T>& sg);

    // b_dot_n is a device pointer [nZnT] (the external-field normal component).
    // Only ever called on full updates (rcosuv/rsinuv are full-update
    // quantities).
    void update(const T* d_bdotn);

    // [nZnT * nThetaEven * nZeta] — regularized normal-derivative kernel,
    // row = source point.
    const T* greenp() const { return greenp_.data(); }
    // [nThetaEven * nZeta] — regularized source term (full surface).
    const T* gstore() const { return gstore_.data(); }

    // [nThetaEven] / [nZeta] — the tan-half-angle tables (golden check).
    const T* tanu() const { return tanu_.data(); }
    const T* tanv() const { return tanv_.data(); }

    int nvper() const { return nvper_; }

   private:
    Sizes sizes_;
    const SurfaceGeometryOperator<T>& sg_;
    cudaStream_t stream_;

    // For an axisymmetric (nZeta == 1) plasma the toroidal direction is
    // resolved by nvper toroidal images rather than the surface grid.
    int nvper_;

    DeviceBuffer<T> tanu_;
    DeviceBuffer<T> tanv_;
    DeviceBuffer<T> tanv_per_;
    DeviceBuffer<T> w_int_;

    DeviceBuffer<T> greenp_;
    DeviceBuffer<T> gstore_;
};

}  // namespace vfield

#endif  // VFIELD_FREE_BOUNDARY_REGULARIZED_INTEGRALS_OPERATOR_HPP_
