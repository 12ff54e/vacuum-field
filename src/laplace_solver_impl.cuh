// laplace_solver_impl.cuh — Laplace-system assembly kernels.
//
// Included once per scalar type by laplace_solver_double.cu /
// laplace_solver_float.cu; see the explicit-instantiation split. Every
// output element is written by exactly one thread replaying vmecpp's
// accumulation order (the g1/g2 staging split keeps the per-mode sums
// bit-identical). The dense factorization and solve run on the host
// (LuSolve, double) — the matrix is tiny.
#ifndef VFIELD_SRC_LAPLACE_SOLVER_IMPL_CUH_
#define VFIELD_SRC_LAPLACE_SOLVER_IMPL_CUH_

#include "vfield/free_boundary/laplace_solver_operator.hpp"

#include <cmath>
#include <numbers>
#include <string>
#include <vector>

namespace vfield {

namespace {

constexpr int kBlockSize = 256;

inline int gridSize(int n) {
    return (n + kBlockSize - 1) / kBlockSize;
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
struct LaplaceKernelParams {
    // Scaled basis tables.
    const T* cosnvScaled;  // [(nf+1) * nZeta]
    const T* sinnvScaled;
    const T* cosmuiScaled;  // [nThetaReduced * (mf+1)]
    const T* sinmuiScaled;
    // Staging.
    T* g1;  // [nZnT * nThetaReduced * (nf+1)] — odd kernel
    T* g2;
    T* g1e;  // [nZnT * nThetaReduced * (nf+1)] — even kernel (lasym)
    T* g2e;
    // Inputs.
    const T* greenp;       // [nZnT * nThetaEven * nZeta]
    const T* gstore;       // [nThetaEven * nZeta]
    const T* singularSin;  // [mnpd * nZnT]
    const T* singularCos;
    const T* singularBvec;  // [mnpd]
    // Sizes.
    int nf;
    int mf;
    int nfp;
    int nZeta;
    int nThetaReduced;
    int nThetaEven;
    int nThetaEff;
    int nZnT;
    int mnpd;
    bool lasym;
    // Outputs.
    T* grpmnSin;  // [mnpd * nZnT]
    T* grpmnCos;
    T* gstoreSymm;  // [nThetaReduced * nZeta]
    T* bcos;        // [(2nf+1) * nThetaReduced]
    T* bsin;
    T* actemp;  // [mnpd * (2nf+1) * nThetaEff]
    T* astemp;
    T* bvecSin;  // [mnpd]
    T* amat;     // [mnpd * mnpd]
    T* matrix;   // [mnpd * mnpd]
    T* bvec;     // [mnpd]
};

// Phase A of transformGreensFunctionDerivative: per (source point, poloidal
// row) compute the toroidal DFTs g1/g2 of the odd/even split of greenp.
template <class T>
__global__ void greenpTransformPhaseAKernel(LaplaceKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int klp = idx / p.nThetaReduced;
    const int l = idx - klp * p.nThetaReduced;
    if (klp >= p.nZnT) return;

    const int full = p.nThetaEven * p.nZeta;
    const int lRev = (p.nThetaEven - l) % p.nThetaEven;

    for (int n = 0; n < p.nf + 1; ++n) {
        T g1 = 0;
        T g2 = 0;
        T g1e = 0;
        T g2e = 0;
        for (int k = 0; k < p.nZeta; ++k) {
            const int kRev = (p.nZeta - k) % p.nZeta;
            const int kl = l * p.nZeta + k;
            const int klRev = lRev * p.nZeta + kRev;
            // 0.5 factor for even/odd decomposition
            const T kernel_odd =
                (p.greenp[klp * full + kl] - p.greenp[klp * full + klRev]) *
                T(0.5);
            g1 += p.cosnvScaled[n * p.nZeta + k] * kernel_odd;
            g2 += p.sinnvScaled[n * p.nZeta + k] * kernel_odd;
            if (p.lasym) {
                const T kernel_even =
                    (p.greenp[klp * full + kl] + p.greenp[klp * full + klRev]) *
                    T(0.5);
                g1e += p.cosnvScaled[n * p.nZeta + k] * kernel_even;
                g2e += p.sinnvScaled[n * p.nZeta + k] * kernel_even;
            }
        }
        p.g1[(klp * p.nThetaReduced + l) * (p.nf + 1) + n] = g1;
        p.g2[(klp * p.nThetaReduced + l) * (p.nf + 1) + n] = g2;
        if (p.lasym) {
            p.g1e[(klp * p.nThetaReduced + l) * (p.nf + 1) + n] = g1e;
            p.g2e[(klp * p.nThetaReduced + l) * (p.nf + 1) + n] = g2e;
        }
    }
}

// Phase B: per (source point, m) accumulate the poloidal analysis of g1/g2
// into the grpmn slots (l ascending, vmecpp's order).
template <class T>
__global__ void greenpTransformPhaseBKernel(LaplaceKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int klp = idx / (p.mf + 1);
    const int m = idx - klp * (p.mf + 1);
    if (klp >= p.nZnT) return;

    for (int n = 0; n < p.nf + 1; ++n) {
        T accum_posn = 0;
        T accum_negn = 0;
        for (int l = 0; l < p.nThetaReduced; ++l) {
            const T cosmui = p.cosmuiScaled[l * (p.mf + 1) + m];
            const T sinmui = p.sinmuiScaled[l * (p.mf + 1) + m];
            const T g1v = p.g1[(klp * p.nThetaReduced + l) * (p.nf + 1) + n];
            const T g2v = p.g2[(klp * p.nThetaReduced + l) * (p.nf + 1) + n];

            const T gcos_symm = g1v * sinmui;
            const T gsin_symm = g2v * cosmui;

            accum_posn += gcos_symm - gsin_symm;
            if (n > 0) { accum_negn += gcos_symm + gsin_symm; }
        }

        const int idx_m_posn = (p.nf + n) * (p.mf + 1) + m;
        p.grpmnSin[idx_m_posn * p.nZnT + klp] = accum_posn;
        if (n > 0) {
            const int idx_m_negn = (p.nf - n) * (p.mf + 1) + m;
            p.grpmnSin[idx_m_negn * p.nZnT + klp] = accum_negn;
        }
    }

    if (p.lasym) {
        // even kernel maps to the cos(mu - nv) and cos(mu + nv) basis
        for (int n = 0; n < p.nf + 1; ++n) {
            T accum_cos_posn = 0;
            T accum_cos_negn = 0;
            for (int l = 0; l < p.nThetaReduced; ++l) {
                const T cosmui = p.cosmuiScaled[l * (p.mf + 1) + m];
                const T sinmui = p.sinmuiScaled[l * (p.mf + 1) + m];
                const T g1v =
                    p.g1e[(klp * p.nThetaReduced + l) * (p.nf + 1) + n];
                const T g2v =
                    p.g2e[(klp * p.nThetaReduced + l) * (p.nf + 1) + n];

                // g1*cosmui + g2*sinmui = cos(mu - nv)  [posn]
                // g1*cosmui - g2*sinmui = cos(mu + nv)  [negn]
                const T gcos_asym = g1v * cosmui;
                const T gsin_asym = g2v * sinmui;

                accum_cos_posn += gcos_asym + gsin_asym;
                if (n > 0) { accum_cos_negn += gcos_asym - gsin_asym; }
            }

            const int idx_m_posn = (p.nf + n) * (p.mf + 1) + m;
            p.grpmnCos[idx_m_posn * p.nZnT + klp] = accum_cos_posn;
            if (n > 0) {
                const int idx_m_negn = (p.nf - n) * (p.mf + 1) + m;
                p.grpmnCos[idx_m_negn * p.nZnT + klp] = accum_cos_negn;
            }
        }
    }
}

// Odd part of the source term under (theta, zeta) -> (-theta, -zeta).
template <class T>
__global__ void symmetriseGstoreKernel(LaplaceKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nThetaReduced * p.nZeta) return;

    const int l = kl / p.nZeta;
    const int k = kl - l * p.nZeta;
    const int lRev = (p.nThetaEven - l) % p.nThetaEven;
    const int kRev = (p.nZeta - k) % p.nZeta;
    const int klRev = lRev * p.nZeta + kRev;

    // 1/2 for even/odd decomposition
    p.gstoreSymm[kl] = (p.gstore[kl] - p.gstore[klRev]) * T(0.5);
}

// grpmn += singular/nfp (full-update accumulation).
template <class T>
__global__ void accumulateFullGrpmnKernel(LaplaceKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.mnpd * p.nZnT) return;

    p.grpmnSin[idx] += p.singularSin[idx] * T(1.0 / p.nfp);
    if (p.lasym) { p.grpmnCos[idx] += p.singularCos[idx] * T(1.0 / p.nfp); }
}

// Phase A of the toroidal transforms: bcos/bsin of the symmetrized source.
template <class T>
__global__ void toroidalFftPhaseAKernel(LaplaceKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = idx / p.nThetaReduced;
    const int l = idx - n * p.nThetaReduced;
    if (n >= p.nf + 1) return;

    T bcos_mat = 0;
    T bsin_mat = 0;
    for (int k = 0; k < p.nZeta; ++k) {
        const T g = p.gstoreSymm[l * p.nZeta + k];
        bcos_mat += p.cosnvScaled[n * p.nZeta + k] * g;
        bsin_mat += p.sinnvScaled[n * p.nZeta + k] * g;
    }

    p.bcos[(p.nf + n) * p.nThetaReduced + l] = bcos_mat;
    p.bsin[(p.nf + n) * p.nThetaReduced + l] = bsin_mat;
    if (n > 0) {
        p.bcos[(p.nf - n) * p.nThetaReduced + l] = bcos_mat;
        p.bsin[(p.nf - n) * p.nThetaReduced + l] = -bsin_mat;
    }
}

// Phase B: toroidal transform of the grpmn kernel (posn slots accumulated,
// negn slots are copies with the astemp sign flip).
template <class T>
__global__ void toroidalFftPhaseBKernel(LaplaceKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int mn = idx / ((p.nf + 1) * p.nThetaEff);
    const int rest = idx - mn * ((p.nf + 1) * p.nThetaEff);
    const int n = rest / p.nThetaEff;
    const int l = rest - n * p.nThetaEff;
    if (mn >= p.mnpd) return;

    T actemp_acc = 0;
    T astemp_acc = 0;
    for (int k = 0; k < p.nZeta; ++k) {
        const T grpmn_val = p.grpmnSin[mn * p.nZnT + l * p.nZeta + k];
        actemp_acc += p.cosnvScaled[n * p.nZeta + k] * grpmn_val;
        astemp_acc += p.sinnvScaled[n * p.nZeta + k] * grpmn_val;
    }

    const int idx_a_posn = (mn * (2 * p.nf + 1) + (p.nf + n)) * p.nThetaEff + l;
    p.actemp[idx_a_posn] = actemp_acc;
    p.astemp[idx_a_posn] = astemp_acc;
    if (n > 0) {
        const int idx_a_negn =
            (mn * (2 * p.nf + 1) + (p.nf - n)) * p.nThetaEff + l;
        p.actemp[idx_a_negn] = actemp_acc;
        p.astemp[idx_a_negn] = -astemp_acc;
    }
}

// Poloidal transform of the source: bvec_sin.
template <class T>
__global__ void poloidalFftBvecKernel(LaplaceKernelParams<T> p) {
    const int mn = blockIdx.x * blockDim.x + threadIdx.x;
    if (mn >= p.mnpd) return;

    const int n = mn / (p.mf + 1);  // 0..2nf (n - nf in -nf..nf)
    const int m = mn % (p.mf + 1);

    T accum = 0;
    for (int l = 0; l < p.nThetaReduced; ++l) {
        accum += p.bcos[n * p.nThetaReduced + l] *
                     p.sinmuiScaled[l * (p.mf + 1) + m] -
                 p.bsin[n * p.nThetaReduced + l] *
                     p.cosmuiScaled[l * (p.mf + 1) + m];
    }
    p.bvecSin[mn] = accum;
}

// Poloidal transform of the kernel: the dense system matrix (column = source
// mode, row = target mode, vmecpp's flat layout).
template <class T>
__global__ void poloidalFftAmatKernel(LaplaceKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int mn = idx / p.mnpd;
    const int row = idx - mn * p.mnpd;
    if (mn >= p.mnpd) return;

    const int m = row % (p.mf + 1);

    T accum = 0;
    for (int l = 0; l < p.nThetaReduced; ++l) {
        const int all_n = row / (p.mf + 1);
        accum += p.actemp[(mn * (2 * p.nf + 1) + all_n) * p.nThetaEff + l] *
                     p.sinmuiScaled[l * (p.mf + 1) + m] -
                 p.astemp[(mn * (2 * p.nf + 1) + all_n) * p.nThetaEff + l] *
                     p.cosmuiScaled[l * (p.mf + 1) + m];
    }
    p.amat[row * p.mnpd + mn] = accum;
}

// matrix = amat (flat copy).
template <class T>
__global__ void buildMatrixCopyKernel(LaplaceKernelParams<T> p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.mnpd * p.mnpd) return;
    p.matrix[i] = p.amat[i];
}

// Gauge rows: zero the (m=0, n in [-nf, 0)) rows across all columns (they
// duplicate the m=0/n>0 sin modes).
template <class T>
__global__ void buildMatrixGaugeKernel(LaplaceKernelParams<T> p) {
    const int mnp = blockIdx.x * blockDim.x + threadIdx.x;
    if (mnp >= p.mnpd) return;
    for (int all_n = 0; all_n < p.nf; ++all_n) {
        p.matrix[mnp * p.mnpd + all_n * (p.mf + 1)] = 0;
    }
}

// +0.5 diagonal (the 1/2*Phi jump term of the second-kind integral
// equation).
template <class T>
__global__ void buildMatrixDiagonalKernel(LaplaceKernelParams<T> p) {
    const int mn = blockIdx.x * blockDim.x + threadIdx.x;
    if (mn >= p.mnpd) return;
    p.matrix[mn * p.mnpd + mn] += T(0.5);
}

// bvec = bvec_sin + singular_bvec/nfp.
template <class T>
__global__ void assembleBvecKernel(LaplaceKernelParams<T> p) {
    const int mn = blockIdx.x * blockDim.x + threadIdx.x;
    if (mn >= p.mnpd) return;
    p.bvec[mn] = p.bvecSin[mn] + p.singularBvec[mn] * T(1.0 / p.nfp);
}

// Gauge RHS entries: zero (m=0, n in [-nf, 0)).
template <class T>
__global__ void gaugeBvecKernel(LaplaceKernelParams<T> p) {
    const int all_n = blockIdx.x * blockDim.x + threadIdx.x;
    if (all_n >= p.nf) return;
    p.bvec[all_n * (p.mf + 1)] = 0;
}

template <class T>
LaplaceSolverOperator<T>::LaplaceSolverOperator(const Sizes& sizes,
                                                const FourierBasisDevice<T>& fb)
    : sizes_(sizes), fb_(fb), stream_(nullptr) {
    nf_ = sizes_.ntor;
    mf_ = sizes_.mpol + 1;
    mnpd_ = (2 * nf_ + 1) * (mf_ + 1);

    // Pre-compute scaled Fourier basis tables (vmecpp's constructor).
    std::vector<double> cosnv_scaled((nf_ + 1) * sizes_.nZeta);
    std::vector<double> sinnv_scaled((nf_ + 1) * sizes_.nZeta);
    FourierBasis host_fb(sizes_);
    for (int n = 0; n < nf_ + 1; ++n) {
        const double scale = 1.0 / host_fb.nscale[n];
        for (int k = 0; k < sizes_.nZeta; ++k) {
            const int idx_nk = n * sizes_.nZeta + k;
            cosnv_scaled[idx_nk] = host_fb.cosnv[idx_nk] * scale;
            sinnv_scaled[idx_nk] = host_fb.sinnv[idx_nk] * scale;
        }
    }
    std::vector<double> cosmui_scaled(sizes_.nThetaReduced * (mf_ + 1));
    std::vector<double> sinmui_scaled(sizes_.nThetaReduced * (mf_ + 1));
    for (int l = 0; l < sizes_.nThetaReduced; ++l) {
        for (int m = 0; m < mf_ + 1; ++m) {
            const int idx_lm = l * (sizes_.mnyq2 + 1) + m;
            const double scale = 1.0 / host_fb.mscale[m];
            cosmui_scaled[l * (mf_ + 1) + m] = host_fb.cosmui[idx_lm] * scale;
            sinmui_scaled[l * (mf_ + 1) + m] = host_fb.sinmui[idx_lm] * scale;
        }
    }

    uploadConverted(cosnv_scaled, &cosnv_scaled_);
    uploadConverted(sinnv_scaled, &sinnv_scaled_);
    uploadConverted(cosmui_scaled, &cosmui_scaled_);
    uploadConverted(sinmui_scaled, &sinmui_scaled_);

    g1_.allocate(sizes_.nZnT * sizes_.nThetaReduced * (nf_ + 1));
    g2_.allocate(sizes_.nZnT * sizes_.nThetaReduced * (nf_ + 1));
    if (sizes_.lasym) {
        g1e_.allocate(sizes_.nZnT * sizes_.nThetaReduced * (nf_ + 1));
        g2e_.allocate(sizes_.nZnT * sizes_.nThetaReduced * (nf_ + 1));
    }

    grpmn_sin_.allocate(mnpd_ * sizes_.nZnT);
    gstore_symm_.allocate(sizes_.nThetaReduced * sizes_.nZeta);
    bcos_.allocate((2 * nf_ + 1) * sizes_.nThetaReduced);
    bsin_.allocate((2 * nf_ + 1) * sizes_.nThetaReduced);
    actemp_.allocate(mnpd_ * (2 * nf_ + 1) * sizes_.nThetaEff);
    astemp_.allocate(mnpd_ * (2 * nf_ + 1) * sizes_.nThetaEff);
    bvec_sin_.allocate(mnpd_);
    amat_.allocate(mnpd_ * mnpd_);
    matrix_.allocate(mnpd_ * mnpd_);
    bvec_.allocate(mnpd_);
    solution_.allocate(mnpd_);
    if (sizes_.lasym) { grpmn_cos_.allocate(mnpd_ * sizes_.nZnT); }

    matrix_h_.assign(mnpd_ * mnpd_, 0.0);
    pivots_h_.assign(mnpd_, 0);
    bvec_h_.assign(mnpd_, 0.0);
}

template <class T>
void LaplaceSolverOperator<T>::transformGreensFunctionDerivative(
    const T* d_greenp) {
    LaplaceKernelParams<T> p{};
    p.cosnvScaled = cosnv_scaled_.data();
    p.sinnvScaled = sinnv_scaled_.data();
    p.cosmuiScaled = cosmui_scaled_.data();
    p.sinmuiScaled = sinmui_scaled_.data();
    p.g1 = g1_.data();
    p.g2 = g2_.data();
    p.g1e = g1e_.data();
    p.g2e = g2e_.data();
    p.greenp = d_greenp;
    p.gstore = nullptr;
    p.singularSin = nullptr;
    p.singularCos = nullptr;
    p.singularBvec = nullptr;
    p.nf = nf_;
    p.mf = mf_;
    p.nZeta = sizes_.nZeta;
    p.nfp = sizes_.nfp;
    p.nThetaReduced = sizes_.nThetaReduced;
    p.nThetaEven = sizes_.nThetaEven;
    p.nThetaEff = sizes_.nThetaEff;
    p.nZnT = sizes_.nZnT;
    p.mnpd = mnpd_;
    p.lasym = sizes_.lasym;
    p.grpmnSin = grpmn_sin_.data();
    p.grpmnCos = grpmn_cos_.data();
    p.gstoreSymm = nullptr;
    p.bcos = nullptr;
    p.bsin = nullptr;
    p.actemp = nullptr;
    p.astemp = nullptr;
    p.bvecSin = nullptr;
    p.amat = nullptr;
    p.matrix = nullptr;
    p.bvec = nullptr;

    launchChecked(
        reinterpret_cast<const void*>(&greenpTransformPhaseAKernel<T>),
        sizes_.nZnT * sizes_.nThetaReduced, kBlockSize, p, stream_,
        "LaplaceSolverOperator: greenp transform phase A");
    launchChecked(
        reinterpret_cast<const void*>(&greenpTransformPhaseBKernel<T>),
        sizes_.nZnT * (mf_ + 1), kBlockSize, p, stream_,
        "LaplaceSolverOperator: greenp transform phase B");
}

template <class T>
void LaplaceSolverOperator<T>::symmetriseSourceTerm(const T* d_gstore) {
    LaplaceKernelParams<T> p{};
    p.cosnvScaled = cosnv_scaled_.data();
    p.sinnvScaled = sinnv_scaled_.data();
    p.cosmuiScaled = cosmui_scaled_.data();
    p.sinmuiScaled = sinmui_scaled_.data();
    p.g1 = nullptr;
    p.g2 = nullptr;
    p.g1e = nullptr;
    p.g2e = nullptr;
    p.greenp = nullptr;
    p.gstore = d_gstore;
    p.singularSin = nullptr;
    p.singularCos = nullptr;
    p.singularBvec = nullptr;
    p.nf = nf_;
    p.mf = mf_;
    p.nZeta = sizes_.nZeta;
    p.nfp = sizes_.nfp;
    p.nThetaReduced = sizes_.nThetaReduced;
    p.nThetaEven = sizes_.nThetaEven;
    p.nThetaEff = sizes_.nThetaEff;
    p.nZnT = sizes_.nZnT;
    p.mnpd = mnpd_;
    p.lasym = sizes_.lasym;
    p.grpmnSin = nullptr;
    p.grpmnCos = nullptr;
    p.gstoreSymm = gstore_symm_.data();
    p.bcos = nullptr;
    p.bsin = nullptr;
    p.actemp = nullptr;
    p.astemp = nullptr;
    p.bvecSin = nullptr;
    p.amat = nullptr;
    p.matrix = nullptr;
    p.bvec = nullptr;

    launchChecked(reinterpret_cast<const void*>(&symmetriseGstoreKernel<T>),
                  sizes_.nThetaReduced * sizes_.nZeta, kBlockSize, p, stream_,
                  "LaplaceSolverOperator: symmetrise source");
}

template <class T>
void LaplaceSolverOperator<T>::accumulateFullGrpmn(const T* d_singular_sin,
                                                   const T* d_singular_cos) {
    LaplaceKernelParams<T> p{};
    p.cosnvScaled = nullptr;
    p.sinnvScaled = nullptr;
    p.cosmuiScaled = nullptr;
    p.sinmuiScaled = nullptr;
    p.g1 = nullptr;
    p.g2 = nullptr;
    p.g1e = nullptr;
    p.g2e = nullptr;
    p.greenp = nullptr;
    p.gstore = nullptr;
    p.singularSin = d_singular_sin;
    p.singularCos = d_singular_cos;
    p.singularBvec = nullptr;
    p.nf = nf_;
    p.mf = mf_;
    p.nZeta = sizes_.nZeta;
    p.nfp = sizes_.nfp;
    p.nThetaReduced = sizes_.nThetaReduced;
    p.nThetaEven = sizes_.nThetaEven;
    p.nThetaEff = sizes_.nThetaEff;
    p.nZnT = sizes_.nZnT;
    p.mnpd = mnpd_;
    p.lasym = sizes_.lasym;
    p.grpmnSin = grpmn_sin_.data();
    p.grpmnCos = grpmn_cos_.data();
    p.gstoreSymm = nullptr;
    p.bcos = nullptr;
    p.bsin = nullptr;
    p.actemp = nullptr;
    p.astemp = nullptr;
    p.bvecSin = nullptr;
    p.amat = nullptr;
    p.matrix = nullptr;
    p.bvec = nullptr;

    launchChecked(reinterpret_cast<const void*>(&accumulateFullGrpmnKernel<T>),
                  mnpd_ * sizes_.nZnT, kBlockSize, p, stream_,
                  "LaplaceSolverOperator: accumulate grpmn");
}

template <class T>
void LaplaceSolverOperator<T>::performToroidalFourierTransforms() {
    LaplaceKernelParams<T> p{};
    p.cosnvScaled = cosnv_scaled_.data();
    p.sinnvScaled = sinnv_scaled_.data();
    p.cosmuiScaled = nullptr;
    p.sinmuiScaled = nullptr;
    p.g1 = nullptr;
    p.g2 = nullptr;
    p.g1e = nullptr;
    p.g2e = nullptr;
    p.greenp = nullptr;
    p.gstore = nullptr;
    p.singularSin = nullptr;
    p.singularCos = nullptr;
    p.singularBvec = nullptr;
    p.nf = nf_;
    p.mf = mf_;
    p.nZeta = sizes_.nZeta;
    p.nfp = sizes_.nfp;
    p.nThetaReduced = sizes_.nThetaReduced;
    p.nThetaEven = sizes_.nThetaEven;
    p.nThetaEff = sizes_.nThetaEff;
    p.nZnT = sizes_.nZnT;
    p.mnpd = mnpd_;
    p.lasym = sizes_.lasym;
    p.grpmnSin = grpmn_sin_.data();
    p.grpmnCos = nullptr;
    p.gstoreSymm = gstore_symm_.data();
    p.bcos = bcos_.data();
    p.bsin = bsin_.data();
    p.actemp = actemp_.data();
    p.astemp = astemp_.data();
    p.bvecSin = nullptr;
    p.amat = nullptr;
    p.matrix = nullptr;
    p.bvec = nullptr;

    launchChecked(reinterpret_cast<const void*>(&toroidalFftPhaseAKernel<T>),
                  (nf_ + 1) * sizes_.nThetaReduced, kBlockSize, p, stream_,
                  "LaplaceSolverOperator: toroidal phase A");
    launchChecked(reinterpret_cast<const void*>(&toroidalFftPhaseBKernel<T>),
                  mnpd_ * (nf_ + 1) * sizes_.nThetaEff, kBlockSize, p, stream_,
                  "LaplaceSolverOperator: toroidal phase B");
}

template <class T>
void LaplaceSolverOperator<T>::performPoloidalFourierTransforms() {
    LaplaceKernelParams<T> p{};
    p.cosnvScaled = nullptr;
    p.sinnvScaled = nullptr;
    p.cosmuiScaled = cosmui_scaled_.data();
    p.sinmuiScaled = sinmui_scaled_.data();
    p.g1 = nullptr;
    p.g2 = nullptr;
    p.g1e = nullptr;
    p.g2e = nullptr;
    p.greenp = nullptr;
    p.gstore = nullptr;
    p.singularSin = nullptr;
    p.singularCos = nullptr;
    p.singularBvec = nullptr;
    p.nf = nf_;
    p.mf = mf_;
    p.nZeta = sizes_.nZeta;
    p.nfp = sizes_.nfp;
    p.nThetaReduced = sizes_.nThetaReduced;
    p.nThetaEven = sizes_.nThetaEven;
    p.nThetaEff = sizes_.nThetaEff;
    p.nZnT = sizes_.nZnT;
    p.mnpd = mnpd_;
    p.lasym = sizes_.lasym;
    p.grpmnSin = nullptr;
    p.grpmnCos = nullptr;
    p.gstoreSymm = nullptr;
    p.bcos = bcos_.data();
    p.bsin = bsin_.data();
    p.actemp = actemp_.data();
    p.astemp = astemp_.data();
    p.bvecSin = bvec_sin_.data();
    p.amat = amat_.data();
    p.matrix = nullptr;
    p.bvec = nullptr;

    launchChecked(reinterpret_cast<const void*>(&poloidalFftBvecKernel<T>),
                  mnpd_, kBlockSize, p, stream_,
                  "LaplaceSolverOperator: poloidal bvec");
    launchChecked(reinterpret_cast<const void*>(&poloidalFftAmatKernel<T>),
                  mnpd_ * mnpd_, kBlockSize, p, stream_,
                  "LaplaceSolverOperator: poloidal amat");
}

template <class T>
void LaplaceSolverOperator<T>::buildMatrix() {
    LaplaceKernelParams<T> p{};
    p.cosnvScaled = nullptr;
    p.sinnvScaled = nullptr;
    p.cosmuiScaled = nullptr;
    p.sinmuiScaled = nullptr;
    p.g1 = nullptr;
    p.g2 = nullptr;
    p.g1e = nullptr;
    p.g2e = nullptr;
    p.greenp = nullptr;
    p.gstore = nullptr;
    p.singularSin = nullptr;
    p.singularCos = nullptr;
    p.singularBvec = nullptr;
    p.nf = nf_;
    p.mf = mf_;
    p.nZeta = sizes_.nZeta;
    p.nfp = sizes_.nfp;
    p.nThetaReduced = sizes_.nThetaReduced;
    p.nThetaEven = sizes_.nThetaEven;
    p.nThetaEff = sizes_.nThetaEff;
    p.nZnT = sizes_.nZnT;
    p.mnpd = mnpd_;
    p.lasym = sizes_.lasym;
    p.grpmnSin = nullptr;
    p.grpmnCos = nullptr;
    p.gstoreSymm = nullptr;
    p.bcos = nullptr;
    p.bsin = nullptr;
    p.actemp = nullptr;
    p.astemp = nullptr;
    p.bvecSin = nullptr;
    p.amat = amat_.data();
    p.matrix = matrix_.data();
    p.bvec = nullptr;

    launchChecked(reinterpret_cast<const void*>(&buildMatrixCopyKernel<T>),
                  mnpd_ * mnpd_, kBlockSize, p, stream_,
                  "LaplaceSolverOperator: matrix copy");
    launchChecked(reinterpret_cast<const void*>(&buildMatrixGaugeKernel<T>),
                  mnpd_, kBlockSize, p, stream_,
                  "LaplaceSolverOperator: matrix gauge");
    launchChecked(reinterpret_cast<const void*>(&buildMatrixDiagonalKernel<T>),
                  mnpd_, kBlockSize, p, stream_,
                  "LaplaceSolverOperator: matrix diagonal");
}

template <class T>
void LaplaceSolverOperator<T>::decomposeMatrix() {
    // The matrix is tiny: copy it to the host and factorize in double
    // (LuSolve, LAPACK dgetrf semantics on the same flat layout vmecpp
    // passed to LAPACK).
    matrix_.download(matrix_h_.data(), matrix_h_.size());
    const int info =
        LuSolve::decompose(matrix_h_.data(), pivots_h_.data(), mnpd_);
    if (info != 0) {
        throw VfieldError(
            "LaplaceSolverOperator::decomposeMatrix: singular "
            "matrix (dgetrf-style info=" +
            std::to_string(info) + ")");
    }
}

template <class T>
void LaplaceSolverOperator<T>::solveForPotential(const T* d_singular_bvec) {
    LaplaceKernelParams<T> p{};
    p.cosnvScaled = nullptr;
    p.sinnvScaled = nullptr;
    p.cosmuiScaled = nullptr;
    p.sinmuiScaled = nullptr;
    p.g1 = nullptr;
    p.g2 = nullptr;
    p.g1e = nullptr;
    p.g2e = nullptr;
    p.greenp = nullptr;
    p.gstore = nullptr;
    p.singularSin = nullptr;
    p.singularCos = nullptr;
    p.singularBvec = d_singular_bvec;
    p.nf = nf_;
    p.mf = mf_;
    p.nZeta = sizes_.nZeta;
    p.nfp = sizes_.nfp;
    p.nThetaReduced = sizes_.nThetaReduced;
    p.nThetaEven = sizes_.nThetaEven;
    p.nThetaEff = sizes_.nThetaEff;
    p.nZnT = sizes_.nZnT;
    p.mnpd = mnpd_;
    p.lasym = sizes_.lasym;
    p.grpmnSin = nullptr;
    p.grpmnCos = nullptr;
    p.gstoreSymm = nullptr;
    p.bcos = nullptr;
    p.bsin = nullptr;
    p.actemp = nullptr;
    p.astemp = nullptr;
    p.bvecSin = bvec_sin_.data();
    p.amat = nullptr;
    p.matrix = nullptr;
    p.bvec = bvec_.data();

    launchChecked(reinterpret_cast<const void*>(&assembleBvecKernel<T>), mnpd_,
                  kBlockSize, p, stream_,
                  "LaplaceSolverOperator: assemble bvec");
    if (nf_ > 0) {
        launchChecked(reinterpret_cast<const void*>(&gaugeBvecKernel<T>), nf_,
                      kBlockSize, p, stream_,
                      "LaplaceSolverOperator: gauge bvec");
    }

    // Solve on the host in double (the potential is control-adjacent; see
    // lu_solve.hpp).
    bvec_.download(bvec_h_.data(), bvec_h_.size());
    LuSolve::solve(matrix_h_.data(), pivots_h_.data(), bvec_h_.data(), mnpd_);
    // Convert to T explicitly (the host solve runs in double).
    std::vector<T> solution_t(mnpd_);
    for (int mn = 0; mn < mnpd_; ++mn) {
        solution_t[mn] = static_cast<T>(bvec_h_[mn]);
    }
    solution_.upload(solution_t.data(), solution_t.size());
}

}  // namespace vfield

#endif  // VFIELD_SRC_LAPLACE_SOLVER_IMPL_CUH_
