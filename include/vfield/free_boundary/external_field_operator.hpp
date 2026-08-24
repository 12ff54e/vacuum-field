// external_field_operator.hpp — external magnetic field on the device.
//
// Port of vmecpp's ExternalMagneticField (free_boundary/external_magnetic_
// field): bilinear (R,Z) interpolation of the mgrid coil field at the LCFS
// points (grid-bounds cropping with the vmecpp warning), plus the
// Hanson-Hirshman axis-current filament field of the net toroidal current
// (factor 1e-7 * 2 * I, Eq. (8) of Hanson & Hirshman 2002), and the covariant
// (b_sub_u/b_sub_v) and normal (b_dot_n, with the minus sign) components. Serial
// over the whole grid. The abscab path is dropped
// (USE_ABSCAB_FOR_AXIS_CURRENT == false in vmecpp).
#ifndef VFIELD_FREE_BOUNDARY_EXTERNAL_FIELD_OPERATOR_HPP_
#define VFIELD_FREE_BOUNDARY_EXTERNAL_FIELD_OPERATOR_HPP_

#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/mgrid_provider.hpp"
#include "vfield/free_boundary/surface_geometry_operator.hpp"
#include "vfield/runtime/device_buffer.cuh"

namespace vfield {

template <class T>
class ExternalFieldOperator {
   public:
    using val_type = T;
    ExternalFieldOperator(const Sizes& sizes,
                          const SurfaceGeometryOperator<T>& sg,
                          const MgridProvider& mgrid);

    // r_axis/z_axis are device pointers over a single field-period module
    // [nZeta]; net_toroidal_current is in Amperes (the VMEC cTor/MU_0 form is
    // undone by the 1e-7 Biot-Savart prefactor).
    void update(const T* d_r_axis, const T* d_z_axis, T net_toroidal_current);

    // [nZnT]
    const T* interp_br() const { return interp_br_.data(); }
    const T* interp_bp() const { return interp_bp_.data(); }
    const T* interp_bz() const { return interp_bz_.data(); }
    const T* curtor_br() const { return curtor_br_.data(); }
    const T* curtor_bp() const { return curtor_bp_.data(); }
    const T* curtor_bz() const { return curtor_bz_.data(); }
    const T* b_sub_u() const { return bsubu_.data(); }
    const T* b_sub_v() const { return bsubv_.data(); }
    const T* b_dot_n() const { return bdotn_.data(); }

    // [3 * (nZeta * nvper + 1)] — the axis polygon in Cartesian coordinates
    // (interleaved x/y/z), closed; exposed for the xpts_axis golden check.
    const T* axis_xyz() const { return axis_xyz_.data(); }

    int nvper() const { return nvper_; }

   private:
    Sizes sizes_;
    const SurfaceGeometryOperator<T>& sg_;
    // For an axisymmetric (nZeta == 1) plasma the toroidal direction is not
    // resolved by the surface grid, so the axis-current filament is replicated
    // over nvper equally-spaced toroidal angles (matching educational_VMEC's
    // nvper = 64 for the tokamak); otherwise nvper is the number of field
    // periods.
    int nvper_;
    cudaStream_t stream_;

    // mgrid grid on device + metadata.
    DeviceBuffer<T> b_r_;
    DeviceBuffer<T> b_p_;
    DeviceBuffer<T> b_z_;
    DeviceBuffer<T> fixed_br_;
    DeviceBuffer<T> fixed_bp_;
    DeviceBuffer<T> fixed_bz_;
    double min_r_;
    double max_r_;
    double delta_r_;
    double min_z_;
    double max_z_;
    double delta_z_;
    int num_r_;
    int num_z_;
    int num_phi_;
    bool has_fixed_field_;

    DeviceBuffer<T> axis_xyz_;
    DeviceBuffer<T> interp_br_;
    DeviceBuffer<T> interp_bp_;
    DeviceBuffer<T> interp_bz_;
    DeviceBuffer<T> curtor_br_;
    DeviceBuffer<T> curtor_bp_;
    DeviceBuffer<T> curtor_bz_;
    DeviceBuffer<T> bsubu_;
    DeviceBuffer<T> bsubv_;
    DeviceBuffer<T> bdotn_;

    // Out-of-bounds flag for the "Plasma Boundary exceeded Vacuum Grid Size"
    // warning (set by the interpolation kernel, checked on the host).
    DeviceBuffer<int> out_of_bounds_;
};

}  // namespace vfield

#endif  // VFIELD_FREE_BOUNDARY_EXTERNAL_FIELD_OPERATOR_HPP_
