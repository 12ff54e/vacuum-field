// vacuum_field_solver.hpp — the free-boundary vacuum-field driver.
//
// Port of vmecpp's Nestor driver (free_boundary/nestor): orchestrates the
// surface geometry, the external field (mgrid + axis current), the
// singular/regularized Green's-function integrals, and the Laplace solve,
// then reconstructs the tangential derivatives of the scalar potential
// (potU/potV), the covariant vacuum field (bSubU/bSubV), the vacuum magnetic
// pressure bSqVac = |B|^2/2 (no mu0, VMEC force convention), the cylindrical
// components B_R/B_phi/B_Z, and the surface-integral scalars
// bSubUVac/bSubVVac. The OpenMP partitioning and the VmecCheckpoint
// machinery are dropped; full_update == (ivacskip == 0).
#ifndef VFIELD_FREE_BOUNDARY_VACUUM_FIELD_SOLVER_HPP_
#define VFIELD_FREE_BOUNDARY_VACUUM_FIELD_SOLVER_HPP_

#include "vfield/common/fourier_basis.hpp"
#include "vfield/common/fourier_basis_device.cuh"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/external_field_operator.hpp"
#include "vfield/free_boundary/laplace_solver_operator.hpp"
#include "vfield/free_boundary/mgrid_provider.hpp"
#include "vfield/free_boundary/regularized_integrals_operator.hpp"
#include "vfield/free_boundary/singular_integrals_operator.hpp"
#include "vfield/free_boundary/surface_geometry_operator.hpp"
#include "vfield/runtime/device_buffer.cuh"

#include <string>
#include <vector>

namespace vfield {

template <class T>
class VacuumFieldSolver {
   public:
    struct Params {
        explicit Params(const Sizes& sizes) : sizes(sizes) {}

        Sizes sizes;
        // mgrid coil currents (A); the file path may be empty when the
        // fixed-field arrays are given instead.
        std::vector<double> coil_currents;
        std::string mgrid_file;
        // Fixed external field at the boundary points [nZnT] (used when
        // mgrid_file is empty; the interpolation is bypassed).
        std::vector<double> fixed_br;
        std::vector<double> fixed_bp;
        std::vector<double> fixed_bz;
    };

    explicit VacuumFieldSolver(const Params& params);

    // Mirrors Nestor::update minus checkpoints/partitioning. The LCFS
    // coefficient arrays are device pointers to mnsize-sized n-major arrays
    // (antisymmetric ones may be nullptr when !lasym); rAxis/zAxis are
    // device pointers over one field-period module [nZeta];
    // netToroidalCurrent is in Amperes (cTor/MU_0); bsubuVac/bsubvVac are
    // host out-scalars (signJ * 2*pi surface integrals of bSubU/bSubV).
    void update(const T* d_rcc,
                const T* d_rss,
                const T* d_rsc,
                const T* d_rcs,
                const T* d_zsc,
                const T* d_zcs,
                const T* d_zcc,
                const T* d_zss,
                int sign_of_jacobian,
                const T* d_r_axis,
                const T* d_z_axis,
                T* bsubu_vac,
                T* bsubv_vac,
                T net_toroidal_current,
                bool full_update);

    // [nZnT] — device-resident outputs.
    const T* potU() const { return potu_.data(); }
    const T* potV() const { return potv_.data(); }
    const T* bSubU() const { return bsubu_.data(); }
    const T* bSubV() const { return bsubv_.data(); }
    const T* bSqVac() const { return bsqvac_.data(); }
    const T* vacuumBR() const { return vacuum_br_.data(); }
    const T* vacuumBPhi() const { return vacuum_bphi_.data(); }
    const T* vacuumBZ() const { return vacuum_bz_.data(); }
    // [mnpd] — the scalar-potential Fourier coefficients.
    const T* potential() const { return ls_.solution(); }

    const SurfaceGeometryOperator<T>& surfaceGeometry() const { return sg_; }
    const ExternalFieldOperator<T>& externalField() const { return ef_; }
    const SingularIntegralsOperator<T>& singularIntegrals() const {
        return si_;
    }
    const RegularizedIntegralsOperator<T>& regularizedIntegrals() const {
        return ri_;
    }
    const LaplaceSolverOperator<T>& laplaceSolver() const { return ls_; }

   private:
    static MgridProvider loadMgrid(const Params& params);

    Sizes sizes_;
    FourierBasis fb_;
    FourierBasisDevice<T> fbd_;
    MgridProvider mgrid_;
    SurfaceGeometryOperator<T> sg_;
    ExternalFieldOperator<T> ef_;
    SingularIntegralsOperator<T> si_;
    RegularizedIntegralsOperator<T> ri_;
    LaplaceSolverOperator<T> ls_;
    cudaStream_t stream_;

    DeviceBuffer<T> w_int_;  // [nThetaEff]

    DeviceBuffer<T> potu_;
    DeviceBuffer<T> potv_;
    DeviceBuffer<T> bsubu_;
    DeviceBuffer<T> bsubv_;
    DeviceBuffer<T> bsqvac_;
    DeviceBuffer<T> vacuum_br_;
    DeviceBuffer<T> vacuum_bphi_;
    DeviceBuffer<T> vacuum_bz_;
    DeviceBuffer<T> surface_integrals_;  // [2]: bSubUVac, bSubVVac
};

}  // namespace vfield

#endif  // VFIELD_FREE_BOUNDARY_VACUUM_FIELD_SOLVER_HPP_
