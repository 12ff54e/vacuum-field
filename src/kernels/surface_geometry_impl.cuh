// kernels/surface_geometry_impl.cuh — LCFS synthesis + derived-quantity kernels.
//
// Included once per scalar type by surface_geometry_double.cu /
// surface_geometry_float.cu; see the explicit-instantiation split. Serial
// over the whole grid (no tangential partitioning): every output element is
// written by exactly one thread, which replays vmecpp's accumulation order
// (n -> m -> k) exactly, so the results are summation-order-identical to the
// CPU reference.
#ifndef VFIELD_SRC_SURFACE_GEOMETRY_IMPL_CUH_
#define VFIELD_SRC_SURFACE_GEOMETRY_IMPL_CUH_

#include "vfield/free_boundary/surface_geometry_operator.hpp"

#include <cmath>
#include <numbers>

namespace vfield {

namespace {

constexpr int BLOCK_SIZE = 256;

inline int gridSize(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// Checked 1D launch of a kernel taking a single param struct by value.
// cudaLaunchKernel copies the params out of the args array at launch time, so
// a stack array is safe.
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

}  // namespace

// All pointers and scalars the surface-geometry kernels need, in one POD
// (the synthesis kernel alone takes ~50 pointers; a single struct keeps the
// launches readable).
template <class T>
struct SurfaceKernelParams {
    // LCFS Fourier coefficients (n-major, mnsize each; antisymmetric ones may
    // be nullptr when !lasym).
    const T* rCC;
    const T* rSS;
    const T* rSC;
    const T* rCS;
    const T* zSC;
    const T* zCS;
    const T* zCC;
    const T* zSS;
    // Fourier basis tables.
    const T* cosmu;
    const T* sinmu;
    const T* cosmum;
    const T* sinmum;
    const T* cosnv;
    const T* sinnv;
    const T* cosnvn;
    const T* sinnvn;
    // Toroidal-angle trig of the surface grid points.
    const T* cosPhi;
    const T* sinPhi;
    // Sizes.
    int mpol;
    int ntor;
    int nZeta;
    int nfp;
    int mnyq2;
    int nThetaReduced;
    int nThetaEven;
    int nZnT;
    bool lasym;
    bool fullUpdate;
    int signOfJacobian;
    // Outputs.
    T* r1b;
    T* z1b;
    T* rcosuv;
    T* rsinuv;
    T* rzb2;
    T* rub;
    T* rvb;
    T* zub;
    T* zvb;
    T* ruu;
    T* ruv;
    T* rvv;
    T* zuu;
    T* zuv;
    T* zvv;
    T* snr;
    T* snv;
    T* snz;
    T* guu;
    T* guv;
    T* gvv;
    T* auu;
    T* auv;
    T* avv;
    T* drv;
    T* r1bAsym;
    T* z1bAsym;
    T* rubAsym;
    T* rvbAsym;
    T* zubAsym;
    T* zvbAsym;
    T* ruuAsym;
    T* ruvAsym;
    T* rvvAsym;
    T* zuuAsym;
    T* zuvAsym;
    T* zvvAsym;
};

// Evaluate the Fourier series over the REDUCED poloidal range [0, pi]
// (nZeta * nThetaReduced points), producing r1b/z1b, the first derivatives
// (and second derivatives on full updates), plus the lasym antisymmetric
// pieces. The second poloidal half is filled by the mirror kernels below.
template <class T>
__global__ void surfaceSynthesisKernel(SurfaceKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZeta * p.nThetaReduced) return;
    const int l = kl / p.nZeta;
    const int k = kl - l * p.nZeta;

    T r1b = 0;
    T z1b = 0;
    T r1b_asym = 0;
    T z1b_asym = 0;
    T rub = 0;
    T rvb = 0;
    T zub = 0;
    T zvb = 0;
    T rub_asym = 0;
    T rvb_asym = 0;
    T zub_asym = 0;
    T zvb_asym = 0;
    T ruu = 0;
    T ruv = 0;
    T rvv = 0;
    T zuu = 0;
    T zuv = 0;
    T zvv = 0;
    T ruu_asym = 0;
    T ruv_asym = 0;
    T rvv_asym = 0;
    T zuu_asym = 0;
    T zuv_asym = 0;
    T zvv_asym = 0;

    for (int n = 0; n < p.ntor + 1; ++n) {
        // needed for second-order toroidal derivatives
        const int nSq = n * p.nfp * n * p.nfp;

        T rmkcc = 0;
        T rmkss = 0;
        T zmksc = 0;
        T zmkcs = 0;
        T rmksc = 0;
        T rmkcs = 0;
        T zmkcc = 0;
        T zmkss = 0;
        T rmkcc_m = 0;
        T rmkcc_mm = 0;
        T rmkss_m = 0;
        T rmkss_mm = 0;
        T zmksc_m = 0;
        T zmksc_mm = 0;
        T zmkcs_m = 0;
        T zmkcs_mm = 0;
        T rmksc_m = 0;
        T rmksc_mm = 0;
        T rmkcs_m = 0;
        T rmkcs_mm = 0;
        T zmkcc_m = 0;
        T zmkcc_mm = 0;
        T zmkss_m = 0;
        T zmkss_mm = 0;

        for (int m = 0; m < p.mpol; ++m) {
            const int idx_mn = n * p.mpol + m;
            const int mSq = m * m;

            const int idx_lm = l * (p.mnyq2 + 1) + m;
            const T cosmu = p.cosmu[idx_lm];
            const T sinmu = p.sinmu[idx_lm];
            const T cosmum = p.cosmum[idx_lm];
            const T sinmum = p.sinmum[idx_lm];
            // second-order poloidal derivatives
            const T cosmumm = -T(mSq) * cosmu;
            const T sinmumm = -T(mSq) * sinmu;

            rmkcc += p.rCC[idx_mn] * cosmu;
            rmkss += p.rSS[idx_mn] * sinmu;
            zmksc += p.zSC[idx_mn] * sinmu;
            zmkcs += p.zCS[idx_mn] * cosmu;

            rmkcc_m += p.rCC[idx_mn] * sinmum;
            rmkcc_mm += p.rCC[idx_mn] * cosmumm;
            rmkss_m += p.rSS[idx_mn] * cosmum;
            rmkss_mm += p.rSS[idx_mn] * sinmumm;
            zmksc_m += p.zSC[idx_mn] * cosmum;
            zmksc_mm += p.zSC[idx_mn] * sinmumm;
            zmkcs_m += p.zCS[idx_mn] * sinmum;
            zmkcs_mm += p.zCS[idx_mn] * cosmumm;

            if (p.lasym) {
                rmksc += p.rSC[idx_mn] * sinmu;
                rmkcs += p.rCS[idx_mn] * cosmu;
                zmkcc += p.zCC[idx_mn] * cosmu;
                zmkss += p.zSS[idx_mn] * sinmu;
                rmksc_m += p.rSC[idx_mn] * cosmum;
                rmksc_mm += p.rSC[idx_mn] * sinmumm;
                rmkcs_m += p.rCS[idx_mn] * sinmum;
                rmkcs_mm += p.rCS[idx_mn] * cosmumm;
                zmkcc_m += p.zCC[idx_mn] * sinmum;
                zmkcc_mm += p.zCC[idx_mn] * cosmumm;
                zmkss_m += p.zSS[idx_mn] * cosmum;
                zmkss_mm += p.zSS[idx_mn] * sinmumm;
            }
        }  // m

        const int idx_nk = n * p.nZeta + k;
        const T cosnv = p.cosnv[idx_nk];
        const T sinnv = p.sinnv[idx_nk];
        const T cosnvn = p.cosnvn[idx_nk];
        const T sinnvn = p.sinnvn[idx_nk];

        r1b += rmkcc * cosnv + rmkss * sinnv;
        z1b += zmksc * cosnv + zmkcs * sinnv;

        rub += rmkcc_m * cosnv + rmkss_m * sinnv;
        rvb += rmkcc * sinnvn + rmkss * cosnvn;
        zub += zmksc_m * cosnv + zmkcs_m * sinnv;
        zvb += zmksc * sinnvn + zmkcs * cosnvn;

        if (p.lasym) {
            r1b_asym += rmksc * cosnv + rmkcs * sinnv;
            z1b_asym += zmkcc * cosnv + zmkss * sinnv;
            rub_asym += rmksc_m * cosnv + rmkcs_m * sinnv;
            rvb_asym += rmksc * sinnvn + rmkcs * cosnvn;
            zub_asym += zmkcc_m * cosnv + zmkss_m * sinnv;
            zvb_asym += zmkcc * sinnvn + zmkss * cosnvn;
        }

        if (p.fullUpdate) {
            // second-order toroidal derivatives
            const T cosnvnn = -T(nSq) * cosnv;
            const T sinnvnn = -T(nSq) * sinnv;

            ruu += rmkcc_mm * cosnv + rmkss_mm * sinnv;
            ruv += rmkcc_m * sinnvn + rmkss_m * cosnvn;
            rvv += rmkcc * cosnvnn + rmkss * sinnvnn;
            zuu += zmksc_mm * cosnv + zmkcs_mm * sinnv;
            zuv += zmksc_m * sinnvn + zmkcs_m * cosnvn;
            zvv += zmksc * cosnvnn + zmkcs * sinnvnn;

            if (p.lasym) {
                ruu_asym += rmksc_mm * cosnv + rmkcs_mm * sinnv;
                ruv_asym += rmksc_m * sinnvn + rmkcs_m * cosnvn;
                rvv_asym += rmksc * cosnvnn + rmkcs * sinnvnn;
                zuu_asym += zmkcc_mm * cosnv + zmkss_mm * sinnv;
                zuv_asym += zmkcc_m * sinnvn + zmkss_m * cosnvn;
                zvv_asym += zmkcc * cosnvnn + zmkss * sinnvnn;
            }
        }
    }  // n

    p.r1b[kl] = r1b;
    p.z1b[kl] = z1b;
    p.rub[kl] = rub;
    p.rvb[kl] = rvb;
    p.zub[kl] = zub;
    p.zvb[kl] = zvb;
    if (p.lasym) {
        p.r1bAsym[kl] = r1b_asym;
        p.z1bAsym[kl] = z1b_asym;
        p.rubAsym[kl] = rub_asym;
        p.rvbAsym[kl] = rvb_asym;
        p.zubAsym[kl] = zub_asym;
        p.zvbAsym[kl] = zvb_asym;
    }
    if (p.fullUpdate) {
        p.ruu[kl] = ruu;
        p.ruv[kl] = ruv;
        p.rvv[kl] = rvv;
        p.zuu[kl] = zuu;
        p.zuv[kl] = zuv;
        p.zvv[kl] = zvv;
        if (p.lasym) {
            p.ruuAsym[kl] = ruu_asym;
            p.ruvAsym[kl] = ruv_asym;
            p.rvvAsym[kl] = rvv_asym;
            p.zuuAsym[kl] = zuu_asym;
            p.zuvAsym[kl] = zuv_asym;
            p.zvvAsym[kl] = zvv_asym;
        }
    }
}

// lasym only: fill the second poloidal half ]pi,2pi[ as the parity-signed
// mirror of (symmetric - antisymmetric) at the reflected point (cf.
// educational_VMEC symrzl). Runs while the first half still holds the pure
// symmetric values.
template <class T>
__global__ void surfaceMirrorLasymSecondHalfKernel(SurfaceKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (p.nThetaReduced - 2) * p.nZeta) return;
    const int l = 1 + idx / p.nZeta;
    const int k = idx - (l - 1) * p.nZeta;
    const int lRev = (p.nThetaEven - l) % p.nThetaEven;
    const int REV = (p.nZeta - k) % p.nZeta;
    const int kl = l * p.nZeta + k;
    const int klRev = lRev * p.nZeta + REV;

    p.r1b[klRev] = p.r1b[kl] - p.r1bAsym[kl];
    p.z1b[klRev] = -p.z1b[kl] + p.z1bAsym[kl];
    p.rub[klRev] = -p.rub[kl] + p.rubAsym[kl];
    p.rvb[klRev] = -p.rvb[kl] + p.rvbAsym[kl];
    p.zub[klRev] = p.zub[kl] - p.zubAsym[kl];
    p.zvb[klRev] = p.zvb[kl] - p.zvbAsym[kl];
    if (p.fullUpdate) {
        p.ruu[klRev] = p.ruu[kl] - p.ruuAsym[kl];
        p.ruv[klRev] = p.ruv[kl] - p.ruvAsym[kl];
        p.rvv[klRev] = p.rvv[kl] - p.rvvAsym[kl];
        p.zuu[klRev] = -p.zuu[kl] + p.zuuAsym[kl];
        p.zuv[klRev] = -p.zuv[kl] + p.zuvAsym[kl];
        p.zvv[klRev] = -p.zvv[kl] + p.zvvAsym[kl];
    }
}

// lasym only: first poloidal half [0,pi] = symmetric + antisymmetric. Must
// run after the second-half mirror.
template <class T>
__global__ void surfaceCombineLasymFirstHalfKernel(SurfaceKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZeta * p.nThetaReduced) return;

    p.r1b[kl] += p.r1bAsym[kl];
    p.z1b[kl] += p.z1bAsym[kl];
    p.rub[kl] += p.rubAsym[kl];
    p.rvb[kl] += p.rvbAsym[kl];
    p.zub[kl] += p.zubAsym[kl];
    p.zvb[kl] += p.zvbAsym[kl];
    if (p.fullUpdate) {
        p.ruu[kl] += p.ruuAsym[kl];
        p.ruv[kl] += p.ruvAsym[kl];
        p.rvv[kl] += p.rvvAsym[kl];
        p.zuu[kl] += p.zuuAsym[kl];
        p.zuv[kl] += p.zuvAsym[kl];
        p.zvv[kl] += p.zvvAsym[kl];
    }
}

// Surface normal (signJ * (-r*N) with N = X_theta x X_phi), the
// NESTOR-normalized metric, and (on full updates) the second fundamental form
// and drv, over the effective poloidal range [0, nZnT).
template <class T>
__global__ void surfaceDerivedKernel(SurfaceKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZnT) return;

    const T sign_j = static_cast<T>(p.signOfJacobian);

    // surface normal vector components
    p.snr[kl] = sign_j * p.r1b[kl] * p.zub[kl];
    p.snv[kl] = sign_j * (p.rub[kl] * p.zvb[kl] - p.zub[kl] * p.rvb[kl]);
    p.snz[kl] = -sign_j * p.r1b[kl] * p.rub[kl];

    // metric elements; used in Imn and Kmn
    p.guu[kl] = p.rub[kl] * p.rub[kl] + p.zub[kl] * p.zub[kl];
    p.guv[kl] =
        2.0 * (p.rub[kl] * p.rvb[kl] + p.zub[kl] * p.zvb[kl]) / T(p.nfp);
    p.gvv[kl] = (p.rvb[kl] * p.rvb[kl] + p.r1b[kl] * p.r1b[kl] +
                 p.zvb[kl] * p.zvb[kl]) /
                (T(p.nfp) * T(p.nfp));

    if (p.fullUpdate) {
        // d^2X/d(ij) . N (used in Kmn)
        p.auu[kl] = (p.ruu[kl] * p.snr[kl] + p.zuu[kl] * p.snz[kl]) / 2;
        p.auv[kl] = (p.ruv[kl] * p.snr[kl] + p.rub[kl] * p.snv[kl] +
                     p.zuv[kl] * p.snz[kl]) /
                    T(p.nfp);
        p.avv[kl] =
            (p.rvb[kl] * p.snv[kl] +
             ((p.rvv[kl] - p.r1b[kl]) * p.snr[kl] + p.zvv[kl] * p.snz[kl]) /
                 2) /
            (T(p.nfp) * T(p.nfp));

        // -(R N^R + Z N^Z)
        p.drv[kl] = -(p.r1b[kl] * p.snr[kl] + p.z1b[kl] * p.snz[kl]);
    }
}

// R^2 + Z^2 over the effective poloidal range (full_update only).
template <class T>
__global__ void surfaceRzb2Kernel(SurfaceKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZnT) return;
    p.rzb2[kl] = p.r1b[kl] * p.r1b[kl] + p.z1b[kl] * p.z1b[kl];
}

// Stellarator-symmetric case: mirror r1b/z1b/rzb2 into the second poloidal
// half (full_update only; runs after surfaceRzb2Kernel).
template <class T>
__global__ void surfaceMirrorSymmetricKernel(SurfaceKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (p.nThetaReduced - 2) * p.nZeta) return;
    const int l = 1 + idx / p.nZeta;
    const int k = idx - (l - 1) * p.nZeta;
    const int lRev = (p.nThetaEven - l) % p.nThetaEven;
    const int REV = (p.nZeta - k) % p.nZeta;
    const int kl = l * p.nZeta + k;
    const int klRev = lRev * p.nZeta + REV;

    p.r1b[klRev] = p.r1b[kl];
    p.z1b[klRev] = -p.z1b[kl];
    p.rzb2[klRev] = p.rzb2[kl];
}

// x = R cos(phi), y = R sin(phi) over the FULL surface (full_update only).
template <class T>
__global__ void surfaceRcosuvKernel(SurfaceKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nThetaEven * p.nZeta) return;
    const int k = kl % p.nZeta;
    p.rcosuv[kl] = p.r1b[kl] * p.cosPhi[k];
    p.rsinuv[kl] = p.r1b[kl] * p.sinPhi[k];
}

template <class T>
SurfaceGeometryOperator<T>::SurfaceGeometryOperator(
    const Sizes& sizes,
    const FourierBasisDevice<T>& fb)
    : sizes_(sizes), fb_(fb), stream_(nullptr) {
    // Host-computed trig tables (field-period angles and grid toroidal
    // angles); uploaded once.
    std::vector<T> cos_per(sizes_.nfp);
    std::vector<T> sin_per(sizes_.nfp);
    const double omega_per = 2.0 * std::numbers::pi / sizes_.nfp;
    for (int p = 0; p < sizes_.nfp; ++p) {
        const double phi_per = omega_per * p;
        cos_per[p] = static_cast<T>(std::cos(phi_per));
        sin_per[p] = static_cast<T>(std::sin(phi_per));
    }
    std::vector<T> cos_phi(sizes_.nZeta);
    std::vector<T> sin_phi(sizes_.nZeta);
    const double omega_phi =
        2.0 * std::numbers::pi / (sizes_.nfp * sizes_.nZeta);
    for (int k = 0; k < sizes_.nZeta; ++k) {
        const double phi = omega_phi * k;
        cos_phi[k] = static_cast<T>(std::cos(phi));
        sin_phi[k] = static_cast<T>(std::sin(phi));
    }

    cos_per_.allocate(sizes_.nfp);
    sin_per_.allocate(sizes_.nfp);
    cos_phi_.allocate(sizes_.nZeta);
    sin_phi_.allocate(sizes_.nZeta);
    cos_per_.upload(cos_per.data(), cos_per.size());
    sin_per_.upload(sin_per.data(), sin_per.size());
    cos_phi_.upload(cos_phi.data(), cos_phi.size());
    sin_phi_.upload(sin_phi.data(), sin_phi.size());

    r1b_.allocate(sizes_.nThetaEven * sizes_.nZeta);
    z1b_.allocate(sizes_.nThetaEven * sizes_.nZeta);
    rcosuv_.allocate(sizes_.nThetaEven * sizes_.nZeta);
    rsinuv_.allocate(sizes_.nThetaEven * sizes_.nZeta);
    rzb2_.allocate(sizes_.nThetaEven * sizes_.nZeta);

    rub_.allocate(sizes_.nZnT);
    rvb_.allocate(sizes_.nZnT);
    zub_.allocate(sizes_.nZnT);
    zvb_.allocate(sizes_.nZnT);
    ruu_.allocate(sizes_.nZnT);
    ruv_.allocate(sizes_.nZnT);
    rvv_.allocate(sizes_.nZnT);
    zuu_.allocate(sizes_.nZnT);
    zuv_.allocate(sizes_.nZnT);
    zvv_.allocate(sizes_.nZnT);

    snr_.allocate(sizes_.nZnT);
    snv_.allocate(sizes_.nZnT);
    snz_.allocate(sizes_.nZnT);
    guu_.allocate(sizes_.nZnT);
    guv_.allocate(sizes_.nZnT);
    gvv_.allocate(sizes_.nZnT);
    auu_.allocate(sizes_.nZnT);
    auv_.allocate(sizes_.nZnT);
    avv_.allocate(sizes_.nZnT);
    drv_.allocate(sizes_.nZnT);

    if (sizes_.lasym) {
        r1b_asym_.allocate(sizes_.nZnT);
        z1b_asym_.allocate(sizes_.nZnT);
        rub_asym_.allocate(sizes_.nZnT);
        rvb_asym_.allocate(sizes_.nZnT);
        zub_asym_.allocate(sizes_.nZnT);
        zvb_asym_.allocate(sizes_.nZnT);
        ruu_asym_.allocate(sizes_.nZnT);
        ruv_asym_.allocate(sizes_.nZnT);
        rvv_asym_.allocate(sizes_.nZnT);
        zuu_asym_.allocate(sizes_.nZnT);
        zuv_asym_.allocate(sizes_.nZnT);
        zvv_asym_.allocate(sizes_.nZnT);
    }
}

template <class T>
void SurfaceGeometryOperator<T>::update(const T* d_rcc,
                                        const T* d_rss,
                                        const T* d_rsc,
                                        const T* d_rcs,
                                        const T* d_zsc,
                                        const T* d_zcs,
                                        const T* d_zcc,
                                        const T* d_zss,
                                        int sign_of_jacobian,
                                        bool full_update) {
    SurfaceKernelParams<T> p{};
    p.rCC = d_rcc;
    p.rSS = d_rss;
    p.rSC = d_rsc;
    p.rCS = d_rcs;
    p.zSC = d_zsc;
    p.zCS = d_zcs;
    p.zCC = d_zcc;
    p.zSS = d_zss;
    p.cosmu = fb_.cosmu();
    p.sinmu = fb_.sinmu();
    p.cosmum = fb_.cosmum();
    p.sinmum = fb_.sinmum();
    p.cosnv = fb_.cosnv();
    p.sinnv = fb_.sinnv();
    p.cosnvn = fb_.cosnvn();
    p.sinnvn = fb_.sinnvn();
    p.cosPhi = cos_phi_.data();
    p.sinPhi = sin_phi_.data();
    p.mpol = sizes_.mpol;
    p.ntor = sizes_.ntor;
    p.nZeta = sizes_.nZeta;
    p.nfp = sizes_.nfp;
    p.mnyq2 = sizes_.mnyq2;
    p.nThetaReduced = sizes_.nThetaReduced;
    p.nThetaEven = sizes_.nThetaEven;
    p.nZnT = sizes_.nZnT;
    p.lasym = sizes_.lasym;
    p.fullUpdate = full_update;
    p.signOfJacobian = sign_of_jacobian;
    p.r1b = r1b_.data();
    p.z1b = z1b_.data();
    p.rcosuv = rcosuv_.data();
    p.rsinuv = rsinuv_.data();
    p.rzb2 = rzb2_.data();
    p.rub = rub_.data();
    p.rvb = rvb_.data();
    p.zub = zub_.data();
    p.zvb = zvb_.data();
    p.ruu = ruu_.data();
    p.ruv = ruv_.data();
    p.rvv = rvv_.data();
    p.zuu = zuu_.data();
    p.zuv = zuv_.data();
    p.zvv = zvv_.data();
    p.snr = snr_.data();
    p.snv = snv_.data();
    p.snz = snz_.data();
    p.guu = guu_.data();
    p.guv = guv_.data();
    p.gvv = gvv_.data();
    p.auu = auu_.data();
    p.auv = auv_.data();
    p.avv = avv_.data();
    p.drv = drv_.data();
    p.r1bAsym = r1b_asym_.data();
    p.z1bAsym = z1b_asym_.data();
    p.rubAsym = rub_asym_.data();
    p.rvbAsym = rvb_asym_.data();
    p.zubAsym = zub_asym_.data();
    p.zvbAsym = zvb_asym_.data();
    p.ruuAsym = ruu_asym_.data();
    p.ruvAsym = ruv_asym_.data();
    p.rvvAsym = rvv_asym_.data();
    p.zuuAsym = zuu_asym_.data();
    p.zuvAsym = zuv_asym_.data();
    p.zvvAsym = zvv_asym_.data();

    const int nReduced = sizes_.nZeta * sizes_.nThetaReduced;
    const int nMirror = (sizes_.nThetaReduced - 2) * sizes_.nZeta;

    launchChecked(reinterpret_cast<const void*>(&surfaceSynthesisKernel<T>),
                  nReduced, BLOCK_SIZE, p, stream_,
                  "SurfaceGeometryOperator::update: synthesis");
    if (sizes_.lasym) {
        launchChecked(reinterpret_cast<const void*>(
                          &surfaceMirrorLasymSecondHalfKernel<T>),
                      nMirror, BLOCK_SIZE, p, stream_,
                      "SurfaceGeometryOperator::update: lasym second half");
        launchChecked(reinterpret_cast<const void*>(
                          &surfaceCombineLasymFirstHalfKernel<T>),
                      nReduced, BLOCK_SIZE, p, stream_,
                      "SurfaceGeometryOperator::update: lasym first half");
    }
    launchChecked(reinterpret_cast<const void*>(&surfaceDerivedKernel<T>),
                  sizes_.nZnT, BLOCK_SIZE, p, stream_,
                  "SurfaceGeometryOperator::update: derived");
    if (full_update) {
        launchChecked(reinterpret_cast<const void*>(&surfaceRzb2Kernel<T>),
                      sizes_.nZnT, BLOCK_SIZE, p, stream_,
                      "SurfaceGeometryOperator::update: rzb2");
        if (!sizes_.lasym) {
            launchChecked(
                reinterpret_cast<const void*>(&surfaceMirrorSymmetricKernel<T>),
                nMirror, BLOCK_SIZE, p, stream_,
                "SurfaceGeometryOperator::update: symmetric mirror");
        }
        launchChecked(reinterpret_cast<const void*>(&surfaceRcosuvKernel<T>),
                      sizes_.nThetaEven * sizes_.nZeta, BLOCK_SIZE, p, stream_,
                      "SurfaceGeometryOperator::update: rcosuv");
    }
}

}  // namespace vfield

#endif  // VFIELD_SRC_SURFACE_GEOMETRY_IMPL_CUH_
