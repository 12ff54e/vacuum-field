// kernels/vacuum_field_solver_impl.cuh — driver kernels: potential reconstruction,
// covariant vacuum field, surface integrals, magnetic pressure.
//
// Included once per scalar type by vacuum_field_solver_double.cu /
// vacuum_field_solver_float.cu; see the explicit-instantiation split. The
// reconstruction replays vmecpp's mn-ascending accumulation order per point;
// the surface integrals run on a single thread (deterministic order).
#ifndef VFIELD_SRC_VACUUM_FIELD_SOLVER_IMPL_CUH_
#define VFIELD_SRC_VACUUM_FIELD_SOLVER_IMPL_CUH_

#include "vfield/free_boundary/vacuum_field_solver.hpp"

#include <cmath>
#include <numbers>
#include <vector>

namespace vfield {

namespace {

constexpr int BLOCK_SIZE = 256;

inline int gridSize(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

template <class KernelParams>
void launchChecked(const void* func,
                   int n,
                   int block,
                   const KernelParams& params,
                   cudaStream_t stream,
                   const char* tag) {
    void* kargs[] = {const_cast<KernelParams*>(&params)};
    check_cuda(cudaLaunchKernel(func, dim3(gridSize(n)), dim3(block), kargs, 0,
                                stream),
               tag);
}

template <class T>
void uploadConverted(const std::vector<double>& src, DeviceBuffer<T>* dst) {
    std::vector<T> converted(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        converted[i] = static_cast<T>(src[i]);
    }
    dst->allocate(src.size());
    dst->upload(converted.data(), converted.size());
}

}  // namespace

template <class T>
struct DriverKernelParams {
    // Fourier basis (raw tables incl. the mscale/nscale normalization).
    const T* cosmu;
    const T* sinmu;
    const T* cosnv;
    const T* sinnv;
    const T* mscale;
    const T* nscale;
    const T* wInt;
    // Potential coefficients (Laplace solution) + external covariant field.
    const T* potential;  // [mnpd]
    const T* efBSubU;    // [nZnT]
    const T* efBSubV;    // [nZnT]
    // Surface geometry.
    const T* r1b;
    const T* rub;
    const T* rvb;
    const T* zub;
    const T* zvb;
    const T* guu;
    const T* guv;
    const T* gvv;
    // Sizes.
    int nf;
    int mf;
    int nZeta;
    int nZnT;
    int nfp;
    int mnyq2;
    int signOfJacobian;
    // Outputs.
    T* potU;
    T* potV;
    T* bSubU;
    T* bSubV;
    T* bSqVac;
    T* vacuumBR;
    T* vacuumBPhi;
    T* vacuumBZ;
    T* surfaceIntegrals;  // [2]
};

// Inverse-DFT of the tangential derivatives of the scalar potential with the
// basis de-normalized by mscale/nscale (vmecpp's reconstruction in
// Nestor::update).
template <class T>
__global__ void potentialReconstructKernel(DriverKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZnT) return;
    const int l = kl / p.nZeta;
    const int k = kl % p.nZeta;

    T potu = 0;
    T potv = 0;
    for (int mn = 0; mn < (2 * p.nf + 1) * (p.mf + 1); ++mn) {
        const int n = mn / (p.mf + 1) - p.nf;  // -nf:nf
        const int m = mn % (p.mf + 1);

        const int abs_n = abs(n);
        const int sign_n = (n > 0) - (n < 0);

        const int idx_lm = l * (p.mnyq2 + 1) + m;
        const T cosmu = p.cosmu[idx_lm] / p.mscale[m];
        const T sinmu = p.sinmu[idx_lm] / p.mscale[m];

        const int idx_nk = abs_n * p.nZeta + k;
        const T cosnv = p.cosnv[idx_nk] / p.nscale[abs_n];
        const T sinnv = p.sinnv[idx_nk] / p.nscale[abs_n];

        const T cos_mu_nv = cosmu * cosnv + T(sign_n) * sinmu * sinnv;

        potu += p.potential[mn] * T(m) * cos_mu_nv;
        potv += p.potential[mn] * T(-n * p.nfp) * cos_mu_nv;
    }  // mn

    p.potU[kl] = potu;
    p.potV[kl] = potv;
}

// Net covariant magnetic field components on the surface: potential +
// external field.
template <class T>
__global__ void bsubKernel(DriverKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZnT) return;
    p.bSubU[kl] = p.potU[kl] + p.efBSubU[kl];
    p.bSubV[kl] = p.potV[kl] + p.efBSubV[kl];
}

// Surface integrals bSubUVac/bSubVVac = signJ * 2*pi * sum_l wInt[l] *
// bSubU/V (single thread, vmecpp's ascending order).
template <class T>
__global__ void bsubSurfIntegralKernel(DriverKernelParams<T> p) {
    T accu = 0;
    T accv = 0;
    for (int kl = 0; kl < p.nZnT; ++kl) {
        const int l = kl / p.nZeta;
        accu += p.bSubU[kl] * p.wInt[l];
        accv += p.bSubV[kl] * p.wInt[l];
    }
    p.surfaceIntegrals[0] =
        T(p.signOfJacobian) * T(2.0 * std::numbers::pi) * accu;
    p.surfaceIntegrals[1] =
        T(p.signOfJacobian) * T(2.0 * std::numbers::pi) * accv;
}

// Vacuum magnetic pressure and cylindrical components: covariant ->
// contravariant via the full-torus metric (guv * nfp/2, gvv * nfp^2),
// bSqVac = |B|^2/2 (no mu0), B_R/B_phi/B_Z from the surface tangents.
template <class T>
__global__ void bsqVacKernel(DriverKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZnT) return;

    // metric elements, without the Nestor-specific normalizations
    const T guu = p.guu[kl];
    const T guv = p.guv[kl] * T(p.nfp) * T(0.5);
    const T gvv = p.gvv[kl] * T(p.nfp) * T(p.nfp);

    const T det = guu * gvv - guv * guv;

    // compute contravariant magnetic field components by inverting the
    // inverse transform (as used in VMEC to go from bContra to bCov)
    const T bsup_u = (gvv * p.bSubU[kl] - guv * p.bSubV[kl]) / det;
    const T bsup_v = (-guv * p.bSubU[kl] + guu * p.bSubV[kl]) / det;

    // magnetic pressure from vacuum: |B|^2/2
    p.bSqVac[kl] = (p.bSubU[kl] * bsup_u + p.bSubV[kl] * bsup_v) * T(0.5);

    // cylindrical components of vacuum magnetic field
    p.vacuumBR[kl] = p.rub[kl] * bsup_u + p.rvb[kl] * bsup_v;
    p.vacuumBPhi[kl] = p.r1b[kl] * bsup_v;
    p.vacuumBZ[kl] = p.zub[kl] * bsup_u + p.zvb[kl] * bsup_v;
}

template <class T>
MgridProvider VacuumFieldSolver<T>::loadMgrid(const Params& params) {
    MgridProvider mgrid;
    if (!params.mgrid_file.empty()) {
        mgrid.loadFile(params.mgrid_file, params.coil_currents);
    } else if (!params.fixed_br.empty()) {
        mgrid.setFixedMagneticField(params.fixed_br, params.fixed_bp,
                                    params.fixed_bz);
    } else {
        throw VfieldError(
            "VacuumFieldSolver: no mgrid file and no fixed "
            "field given");
    }
    return mgrid;
}

template <class T>
VacuumFieldSolver<T>::VacuumFieldSolver(const Params& params)
    : sizes_(params.sizes),
      fb_(params.sizes),
      fbd_(fb_, params.sizes.lasym, params.sizes.nThetaEven),
      // The external-field operator needs the mgrid loaded before its
      // construction; mgrid_ is declared before ef_, so the static loader
      // runs first.
      mgrid_(loadMgrid(params)),
      sg_(params.sizes, fbd_),
      ef_(params.sizes, sg_, mgrid_),
      si_(params.sizes, fbd_, sg_),
      ri_(params.sizes, sg_),
      ls_(params.sizes, fbd_),
      stream_(nullptr) {
    uploadConverted(sizes_.wInt, &w_int_);

    potu_.allocate(sizes_.nZnT);
    potv_.allocate(sizes_.nZnT);
    bsubu_.allocate(sizes_.nZnT);
    bsubv_.allocate(sizes_.nZnT);
    bsqvac_.allocate(sizes_.nZnT);
    vacuum_br_.allocate(sizes_.nZnT);
    vacuum_bphi_.allocate(sizes_.nZnT);
    vacuum_bz_.allocate(sizes_.nZnT);
    surface_integrals_.allocate(2);
}

template <class T>
void VacuumFieldSolver<T>::update(const T* d_rcc,
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
                                  bool full_update) {
    sg_.update(d_rcc, d_rss, d_rsc, d_rcs, d_zsc, d_zcs, d_zcc, d_zss,
               sign_of_jacobian, full_update);

    ef_.update(d_r_axis, d_z_axis, net_toroidal_current);

    si_.update(ef_.bDotN(), full_update);

    if (full_update) {
        ri_.update(ef_.bDotN());
        ls_.transformGreensFunctionDerivative(ri_.greenp());
        ls_.symmetriseSourceTerm(ri_.gstore());
        ls_.accumulateFullGrpmn(si_.grpmnSin(), si_.grpmnCos());
        ls_.performToroidalFourierTransforms();
        ls_.performPoloidalFourierTransforms();
        ls_.buildMatrix();
        ls_.decomposeMatrix();
    }

    ls_.solveForPotential(si_.bvecSin());

    DriverKernelParams<T> p{};
    p.cosmu = fbd_.cosmu();
    p.sinmu = fbd_.sinmu();
    p.cosnv = fbd_.cosnv();
    p.sinnv = fbd_.sinnv();
    p.mscale = fbd_.mscale();
    p.nscale = fbd_.nscale();
    p.wInt = w_int_.data();
    p.potential = ls_.solution();
    p.efBSubU = ef_.bSubU();
    p.efBSubV = ef_.bSubV();
    p.r1b = sg_.r1b();
    p.rub = sg_.rub();
    p.rvb = sg_.rvb();
    p.zub = sg_.zub();
    p.zvb = sg_.zvb();
    p.guu = sg_.guu();
    p.guv = sg_.guv();
    p.gvv = sg_.gvv();
    p.nf = sizes_.ntor;
    p.mf = sizes_.mpol + 1;
    p.nZeta = sizes_.nZeta;
    p.nZnT = sizes_.nZnT;
    p.nfp = sizes_.nfp;
    p.mnyq2 = sizes_.mnyq2;
    p.signOfJacobian = sign_of_jacobian;
    p.potU = potu_.data();
    p.potV = potv_.data();
    p.bSubU = bsubu_.data();
    p.bSubV = bsubv_.data();
    p.bSqVac = bsqvac_.data();
    p.vacuumBR = vacuum_br_.data();
    p.vacuumBPhi = vacuum_bphi_.data();
    p.vacuumBZ = vacuum_bz_.data();
    p.surfaceIntegrals = surface_integrals_.data();

    launchChecked(reinterpret_cast<const void*>(&potentialReconstructKernel<T>),
                  sizes_.nZnT, BLOCK_SIZE, p, stream_,
                  "VacuumFieldSolver::update: potential reconstruction");
    launchChecked(reinterpret_cast<const void*>(&bsubKernel<T>), sizes_.nZnT,
                  BLOCK_SIZE, p, stream_, "VacuumFieldSolver::update: bsub");
    launchChecked(reinterpret_cast<const void*>(&bsubSurfIntegralKernel<T>), 1,
                  1, p, stream_,
                  "VacuumFieldSolver::update: surface integrals");
    launchChecked(reinterpret_cast<const void*>(&bsqVacKernel<T>), sizes_.nZnT,
                  BLOCK_SIZE, p, stream_, "VacuumFieldSolver::update: bsqvac");

    // Download the surface-integral scalars (host out-parameters).
    T integrals[2];
    surface_integrals_.download(integrals, 2);
    *bsubu_vac = integrals[0];
    *bsubv_vac = integrals[1];
}

}  // namespace vfield

#endif  // VFIELD_SRC_VACUUM_FIELD_SOLVER_IMPL_CUH_
