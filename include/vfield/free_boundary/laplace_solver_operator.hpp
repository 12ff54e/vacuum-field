// laplace_solver_operator.hpp — Laplace-system assembly and solve.
//
// Port of vmecpp's LaplaceSolver (free_boundary/laplace_solver): the
// sin(m*theta - n*zeta) Fourier analysis of the regularized Green's
// function (grpmn), the odd-symmetrized source term, the toroidal and
// poloidal DFTs that assemble the dense system, the gauge row/column
// zeroing of the duplicated m=0/n<0 modes with the +0.5 diagonal (the
// 1/2*Phi jump term of the second-kind integral equation), and the dense
// solve. The matrix and the potential solve live on the host (LuSolve, in
// double — the system is mnpd = (2*ntor+1)(mpol+2) <= ~300); everything
// else runs on the device with vmecpp's exact accumulation order.
// nf = ntor, mf = mpol + 1, mnpd = (2*nf+1)(mf+1).
#ifndef VFIELD_FREE_BOUNDARY_LAPLACE_SOLVER_OPERATOR_HPP_
#define VFIELD_FREE_BOUNDARY_LAPLACE_SOLVER_OPERATOR_HPP_

#include "vfield/common/fourier_basis_device.cuh"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/lu_solve.hpp"
#include "vfield/runtime/device_buffer.cuh"

#include <vector>

namespace vfield {

template <class T>
class LaplaceSolverOperator {
   public:
    LaplaceSolverOperator(const Sizes& sizes, const FourierBasisDevice<T>& fb);

    // The analysis pipeline (in vmecpp's order):
    //   transformGreensFunctionDerivative(greenp) -> grpmn (regularized part)
    //   symmetriseSourceTerm(gstore) -> gstore_symm (odd part)
    //   accumulateFullGrpmn(singular grpmn) += singular / nfp
    //   performToroidalFourierTransforms() -> bcos/bsin + actemp/astemp
    //   performPoloidalFourierTransforms() -> bvec_sin + amat
    //   buildMatrix() -> matrix (gauge rows zeroed, +0.5 diagonal)
    //   decomposeMatrix() -> host LU factorization (full updates only)
    //   solveForPotential(singular bvec) -> solution (potential coeffs)
    void transformGreensFunctionDerivative(const T* d_greenp);
    void symmetriseSourceTerm(const T* d_gstore);
    void accumulateFullGrpmn(const T* d_singular_sin, const T* d_singular_cos);
    void performToroidalFourierTransforms();
    void performPoloidalFourierTransforms();
    void buildMatrix();
    void decomposeMatrix();
    void solveForPotential(const T* d_singular_bvec);

    // [mnpd * nZnT] — full grpmn (regularized + singular/nfp).
    const T* grpmnSin() const { return grpmn_sin_.data(); }
    const T* grpmnCos() const { return grpmn_cos_.data(); }
    // [nThetaReduced * nZeta] — odd-symmetrized source.
    const T* gstoreSymm() const { return gstore_symm_.data(); }
    // [(2nf+1) * nThetaReduced].
    const T* bcos() const { return bcos_.data(); }
    const T* bsin() const { return bsin_.data(); }
    // [mnpd * (2nf+1) * nThetaEff].
    const T* actemp() const { return actemp_.data(); }
    const T* astemp() const { return astemp_.data(); }
    // [mnpd] — the regularized part of the RHS.
    const T* bvecSin() const { return bvec_sin_.data(); }
    // [mnpd * mnpd] — the assembled matrix (flat, vmecpp layout).
    const T* matrix() const { return matrix_.data(); }
    // [mnpd] — the assembled RHS / the potential solution.
    const T* bvec() const { return bvec_.data(); }
    const T* solution() const { return solution_.data(); }

    int nf() const { return nf_; }
    int mf() const { return mf_; }

   private:
    Sizes sizes_;
    const FourierBasisDevice<T>& fb_;
    cudaStream_t stream_;

    int nf_;
    int mf_;
    int mnpd_;

    // Scaled basis tables (host-precomputed, uploaded).
    DeviceBuffer<T> cosnv_scaled_;  // [(nf+1) * nZeta]
    DeviceBuffer<T> sinnv_scaled_;
    DeviceBuffer<T> cosmui_scaled_;  // [nThetaReduced * (mf+1)]
    DeviceBuffer<T> sinmui_scaled_;

    // Two-pass staging for the greenp analysis: g1/g2 (odd kernel) and
    // g1e/g2e (even kernel, lasym) per (klp, l).
    DeviceBuffer<T> g1_;  // [nZnT * nThetaReduced * (nf+1)]
    DeviceBuffer<T> g2_;
    DeviceBuffer<T> g1e_;
    DeviceBuffer<T> g2e_;

    DeviceBuffer<T> grpmn_sin_;  // [mnpd * nZnT]
    DeviceBuffer<T> grpmn_cos_;
    DeviceBuffer<T> gstore_symm_;  // [nThetaReduced * nZeta]
    DeviceBuffer<T> bcos_;         // [(2nf+1) * nThetaReduced]
    DeviceBuffer<T> bsin_;
    DeviceBuffer<T> actemp_;  // [mnpd * (2nf+1) * nThetaEff]
    DeviceBuffer<T> astemp_;
    DeviceBuffer<T> bvec_sin_;  // [mnpd]
    DeviceBuffer<T> amat_;      // [mnpd * mnpd]
    DeviceBuffer<T> matrix_;    // [mnpd * mnpd]
    DeviceBuffer<T> bvec_;      // [mnpd]
    DeviceBuffer<T> solution_;  // [mnpd]

    // Host copies for the dense solve (always double).
    std::vector<double> matrix_h_;
    std::vector<int> pivots_h_;
    std::vector<double> bvec_h_;
};

}  // namespace vfield

#endif  // VFIELD_FREE_BOUNDARY_LAPLACE_SOLVER_OPERATOR_HPP_
