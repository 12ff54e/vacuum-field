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

inline int grid_size(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

template <class KernelParams>
void launch_checked(const void* func,
                   int n,
                   int block,
                   const KernelParams& params,
                   cudaStream_t stream,
                   const char* tag) {
    void* kargs[] = {const_cast<KernelParams*>(&params)};
    check_cuda(cudaLaunchKernel(func, dim3(grid_size(n)), dim3(block), kargs, 0,
                                stream),
               tag);
}

template <class T>
void upload_converted(const std::vector<double>& src, DeviceBuffer<T>* dst) {
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
    using val_type = T;
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
    const T* ef_b_sub_u;    // [nZnT]
    const T* ef_b_sub_v;    // [nZnT]
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
    int sign_of_jacobian;
    // Outputs.
    T* pot_u;
    T* pot_v;
    T* b_sub_u;
    T* b_sub_v;
    T* b_sq_vac;
    T* vacuum_b_r;
    T* vacuum_b_phi;
    T* vacuum_b_z;
    T* surface_integrals;  // [2]
};

// Inverse-DFT of the tangential derivatives of the scalar potential with the
// basis de-normalized by mscale/nscale (vmecpp's reconstruction in
// Nestor::update).
template <class T>
__global__ void potential_reconstruct_kernel(DriverKernelParams<T> p) {
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

    p.pot_u[kl] = potu;
    p.pot_v[kl] = potv;
}

// Net covariant magnetic field components on the surface: potential +
// external field.
template <class T>
__global__ void bsub_kernel(DriverKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZnT) return;
    p.b_sub_u[kl] = p.pot_u[kl] + p.ef_b_sub_u[kl];
    p.b_sub_v[kl] = p.pot_v[kl] + p.ef_b_sub_v[kl];
}

// Surface integrals b_sub_u_vac/b_sub_v_vac = signJ * 2*pi * sum_l wInt[l] *
// b_sub_u/V (single thread, vmecpp's ascending order).
template <class T>
__global__ void bsub_surf_integral_kernel(DriverKernelParams<T> p) {
    T accu = 0;
    T accv = 0;
    for (int kl = 0; kl < p.nZnT; ++kl) {
        const int l = kl / p.nZeta;
        accu += p.b_sub_u[kl] * p.wInt[l];
        accv += p.b_sub_v[kl] * p.wInt[l];
    }
    p.surface_integrals[0] =
        T(p.sign_of_jacobian) * T(2.0 * std::numbers::pi) * accu;
    p.surface_integrals[1] =
        T(p.sign_of_jacobian) * T(2.0 * std::numbers::pi) * accv;
}

// Vacuum magnetic pressure and cylindrical components: covariant ->
// contravariant via the full-torus metric (guv * nfp/2, gvv * nfp^2),
// b_sq_vac = |B|^2/2 (no mu0), B_R/B_phi/B_Z from the surface tangents.
template <class T>
__global__ void bsq_vac_kernel(DriverKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZnT) return;

    // metric elements, without the Nestor-specific normalizations
    const T guu = p.guu[kl];
    const T guv = p.guv[kl] * T(p.nfp) * T(0.5);
    const T gvv = p.gvv[kl] * T(p.nfp) * T(p.nfp);

    const T det = guu * gvv - guv * guv;

    // compute contravariant magnetic field components by inverting the
    // inverse transform (as used in VMEC to go from b_contra to b_cov)
    const T bsup_u = (gvv * p.b_sub_u[kl] - guv * p.b_sub_v[kl]) / det;
    const T bsup_v = (-guv * p.b_sub_u[kl] + guu * p.b_sub_v[kl]) / det;

    // magnetic pressure from vacuum: |B|^2/2
    p.b_sq_vac[kl] = (p.b_sub_u[kl] * bsup_u + p.b_sub_v[kl] * bsup_v) * T(0.5);

    // cylindrical components of vacuum magnetic field
    p.vacuum_b_r[kl] = p.rub[kl] * bsup_u + p.rvb[kl] * bsup_v;
    p.vacuum_b_phi[kl] = p.r1b[kl] * bsup_v;
    p.vacuum_b_z[kl] = p.zub[kl] * bsup_u + p.zvb[kl] * bsup_v;
}

template <class T>
MgridProvider VacuumFieldSolver<T>::load_mgrid(const Params& params) {
    MgridProvider mgrid;
    if (!params.mgrid_file.empty()) {
        mgrid.load_file(params.mgrid_file, params.coil_currents);
    } else if (!params.fixed_br.empty()) {
        mgrid.set_fixed_magnetic_field(params.fixed_br, params.fixed_bp,
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
      mgrid_(load_mgrid(params)),
      sg_(params.sizes, fbd_),
      ef_(params.sizes, sg_, mgrid_),
      si_(params.sizes, fbd_, sg_),
      ri_(params.sizes, sg_),
      ls_(params.sizes, fbd_),
      stream_(nullptr) {
    upload_converted(sizes_.wInt, &w_int_);

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

    si_.update(ef_.b_dot_n(), full_update);

    if (full_update) {
        ri_.update(ef_.b_dot_n());
        ls_.transform_greens_function_derivative(ri_.greenp());
        ls_.symmetrise_source_term(ri_.gstore());
        ls_.accumulate_full_grpmn(si_.grpmn_sin(), si_.grpmn_cos());
        ls_.perform_toroidal_fourier_transforms();
        ls_.perform_poloidal_fourier_transforms();
        ls_.build_matrix();
        ls_.decompose_matrix();
    }

    ls_.solve_for_potential(si_.bvec_sin());

    DriverKernelParams<T> p{};
    p.cosmu = fbd_.cosmu();
    p.sinmu = fbd_.sinmu();
    p.cosnv = fbd_.cosnv();
    p.sinnv = fbd_.sinnv();
    p.mscale = fbd_.mscale();
    p.nscale = fbd_.nscale();
    p.wInt = w_int_.data();
    p.potential = ls_.solution();
    p.ef_b_sub_u = ef_.b_sub_u();
    p.ef_b_sub_v = ef_.b_sub_v();
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
    p.sign_of_jacobian = sign_of_jacobian;
    p.pot_u = potu_.data();
    p.pot_v = potv_.data();
    p.b_sub_u = bsubu_.data();
    p.b_sub_v = bsubv_.data();
    p.b_sq_vac = bsqvac_.data();
    p.vacuum_b_r = vacuum_br_.data();
    p.vacuum_b_phi = vacuum_bphi_.data();
    p.vacuum_b_z = vacuum_bz_.data();
    p.surface_integrals = surface_integrals_.data();

    launch_checked(reinterpret_cast<const void*>(&potential_reconstruct_kernel<T>),
                  sizes_.nZnT, BLOCK_SIZE, p, stream_,
                  "VacuumFieldSolver::update: potential reconstruction");
    launch_checked(reinterpret_cast<const void*>(&bsub_kernel<T>), sizes_.nZnT,
                  BLOCK_SIZE, p, stream_, "VacuumFieldSolver::update: bsub");
    launch_checked(reinterpret_cast<const void*>(&bsub_surf_integral_kernel<T>), 1,
                  1, p, stream_,
                  "VacuumFieldSolver::update: surface integrals");
    launch_checked(reinterpret_cast<const void*>(&bsq_vac_kernel<T>), sizes_.nZnT,
                  BLOCK_SIZE, p, stream_, "VacuumFieldSolver::update: bsqvac");

    // Download the surface-integral scalars (host out-parameters).
    T integrals[2];
    surface_integrals_.download(integrals, 2);
    *bsubu_vac = integrals[0];
    *bsubv_vac = integrals[1];
}

}  // namespace vfield

#endif  // VFIELD_SRC_VACUUM_FIELD_SOLVER_IMPL_CUH_
