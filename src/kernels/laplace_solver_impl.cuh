// kernels/laplace_solver_impl.cuh — Laplace-system assembly kernels.
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
                   std::string_view tag) {
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
struct LaplaceKernelParams {
    using val_type = T;
    // Scaled basis tables.
    const T* cosnv_scaled;  // [(nf+1) * nZeta]
    const T* sinnv_scaled;
    const T* cosmui_scaled;  // [nThetaReduced * (mf+1)]
    const T* sinmui_scaled;
    // Staging.
    T* g1;  // [nZnT * nThetaReduced * (nf+1)] — odd kernel
    T* g2;
    T* g1e;  // [nZnT * nThetaReduced * (nf+1)] — even kernel (lasym)
    T* g2e;
    // Inputs.
    const T* greenp;       // [nZnT * nThetaEven * nZeta]
    const T* gstore;       // [nThetaEven * nZeta]
    const T* singular_sin;  // [mnpd * nZnT]
    const T* singular_cos;
    const T* singular_bvec;  // [mnpd]
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
    T* grpmn_sin;  // [mnpd * nZnT]
    T* grpmn_cos;
    T* gstore_symm;  // [nThetaReduced * nZeta]
    T* bcos;        // [(2nf+1) * nThetaReduced]
    T* bsin;
    T* actemp;  // [mnpd * (2nf+1) * nThetaEff]
    T* astemp;
    T* bvec_sin;  // [mnpd]
    T* amat;     // [mnpd * mnpd]
    T* matrix;   // [mnpd * mnpd]
    T* bvec;     // [mnpd]
};

// Phase A of transform_greens_function_derivative: per (source point, poloidal
// row) compute the toroidal DFTs g1/g2 of the odd/even split of greenp.
template <class T>
__global__ void greenp_transform_phase_a_kernel(LaplaceKernelParams<T> p) {
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
            const int REV = (p.nZeta - k) % p.nZeta;
            const int kl = l * p.nZeta + k;
            const int klRev = lRev * p.nZeta + REV;
            // 0.5 factor for even/odd decomposition
            const T kernel_odd =
                (p.greenp[klp * full + kl] - p.greenp[klp * full + klRev]) *
                T(0.5);
            g1 += p.cosnv_scaled[n * p.nZeta + k] * kernel_odd;
            g2 += p.sinnv_scaled[n * p.nZeta + k] * kernel_odd;
            if (p.lasym) {
                const T kernel_even =
                    (p.greenp[klp * full + kl] + p.greenp[klp * full + klRev]) *
                    T(0.5);
                g1e += p.cosnv_scaled[n * p.nZeta + k] * kernel_even;
                g2e += p.sinnv_scaled[n * p.nZeta + k] * kernel_even;
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
__global__ void greenp_transform_phase_b_kernel(LaplaceKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int klp = idx / (p.mf + 1);
    const int m = idx - klp * (p.mf + 1);
    if (klp >= p.nZnT) return;

    for (int n = 0; n < p.nf + 1; ++n) {
        T accum_posn = 0;
        T accum_negn = 0;
        for (int l = 0; l < p.nThetaReduced; ++l) {
            const T cosmui = p.cosmui_scaled[l * (p.mf + 1) + m];
            const T sinmui = p.sinmui_scaled[l * (p.mf + 1) + m];
            const T g1v = p.g1[(klp * p.nThetaReduced + l) * (p.nf + 1) + n];
            const T g2v = p.g2[(klp * p.nThetaReduced + l) * (p.nf + 1) + n];

            const T gcos_symm = g1v * sinmui;
            const T gsin_symm = g2v * cosmui;

            accum_posn += gcos_symm - gsin_symm;
            if (n > 0) { accum_negn += gcos_symm + gsin_symm; }
        }

        const int idx_m_posn = (p.nf + n) * (p.mf + 1) + m;
        p.grpmn_sin[idx_m_posn * p.nZnT + klp] = accum_posn;
        if (n > 0) {
            const int idx_m_negn = (p.nf - n) * (p.mf + 1) + m;
            p.grpmn_sin[idx_m_negn * p.nZnT + klp] = accum_negn;
        }
    }

    if (p.lasym) {
        // even kernel maps to the cos(mu - nv) and cos(mu + nv) basis
        for (int n = 0; n < p.nf + 1; ++n) {
            T accum_cos_posn = 0;
            T accum_cos_negn = 0;
            for (int l = 0; l < p.nThetaReduced; ++l) {
                const T cosmui = p.cosmui_scaled[l * (p.mf + 1) + m];
                const T sinmui = p.sinmui_scaled[l * (p.mf + 1) + m];
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
            p.grpmn_cos[idx_m_posn * p.nZnT + klp] = accum_cos_posn;
            if (n > 0) {
                const int idx_m_negn = (p.nf - n) * (p.mf + 1) + m;
                p.grpmn_cos[idx_m_negn * p.nZnT + klp] = accum_cos_negn;
            }
        }
    }
}

// Odd part of the source term under (theta, zeta) -> (-theta, -zeta).
template <class T>
__global__ void symmetrise_gstore_kernel(LaplaceKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nThetaReduced * p.nZeta) return;

    const int l = kl / p.nZeta;
    const int k = kl - l * p.nZeta;
    const int lRev = (p.nThetaEven - l) % p.nThetaEven;
    const int REV = (p.nZeta - k) % p.nZeta;
    const int klRev = lRev * p.nZeta + REV;

    // 1/2 for even/odd decomposition
    p.gstore_symm[kl] = (p.gstore[kl] - p.gstore[klRev]) * T(0.5);
}

// grpmn += singular/nfp (full-update accumulation).
template <class T>
__global__ void accumulate_full_grpmn_kernel(LaplaceKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.mnpd * p.nZnT) return;

    p.grpmn_sin[idx] += p.singular_sin[idx] * T(1.0 / p.nfp);
    if (p.lasym) { p.grpmn_cos[idx] += p.singular_cos[idx] * T(1.0 / p.nfp); }
}

// Phase A of the toroidal transforms: bcos/bsin of the symmetrized source.
template <class T>
__global__ void toroidal_fft_phase_a_kernel(LaplaceKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = idx / p.nThetaReduced;
    const int l = idx - n * p.nThetaReduced;
    if (n >= p.nf + 1) return;

    T bcos_mat = 0;
    T bsin_mat = 0;
    for (int k = 0; k < p.nZeta; ++k) {
        const T g = p.gstore_symm[l * p.nZeta + k];
        bcos_mat += p.cosnv_scaled[n * p.nZeta + k] * g;
        bsin_mat += p.sinnv_scaled[n * p.nZeta + k] * g;
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
__global__ void toroidal_fft_phase_b_kernel(LaplaceKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int mn = idx / ((p.nf + 1) * p.nThetaEff);
    const int rest = idx - mn * ((p.nf + 1) * p.nThetaEff);
    const int n = rest / p.nThetaEff;
    const int l = rest - n * p.nThetaEff;
    if (mn >= p.mnpd) return;

    T actemp_acc = 0;
    T astemp_acc = 0;
    for (int k = 0; k < p.nZeta; ++k) {
        const T grpmn_val = p.grpmn_sin[mn * p.nZnT + l * p.nZeta + k];
        actemp_acc += p.cosnv_scaled[n * p.nZeta + k] * grpmn_val;
        astemp_acc += p.sinnv_scaled[n * p.nZeta + k] * grpmn_val;
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
__global__ void poloidal_fft_bvec_kernel(LaplaceKernelParams<T> p) {
    const int mn = blockIdx.x * blockDim.x + threadIdx.x;
    if (mn >= p.mnpd) return;

    const int n = mn / (p.mf + 1);  // 0..2nf (n - nf in -nf..nf)
    const int m = mn % (p.mf + 1);

    T accum = 0;
    for (int l = 0; l < p.nThetaReduced; ++l) {
        accum += p.bcos[n * p.nThetaReduced + l] *
                     p.sinmui_scaled[l * (p.mf + 1) + m] -
                 p.bsin[n * p.nThetaReduced + l] *
                     p.cosmui_scaled[l * (p.mf + 1) + m];
    }
    p.bvec_sin[mn] = accum;
}

// Poloidal transform of the kernel: the dense system matrix (column = source
// mode, row = target mode, vmecpp's flat layout).
template <class T>
__global__ void poloidal_fft_amat_kernel(LaplaceKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int mn = idx / p.mnpd;
    const int row = idx - mn * p.mnpd;
    if (mn >= p.mnpd) return;

    const int m = row % (p.mf + 1);

    T accum = 0;
    for (int l = 0; l < p.nThetaReduced; ++l) {
        const int all_n = row / (p.mf + 1);
        accum += p.actemp[(mn * (2 * p.nf + 1) + all_n) * p.nThetaEff + l] *
                     p.sinmui_scaled[l * (p.mf + 1) + m] -
                 p.astemp[(mn * (2 * p.nf + 1) + all_n) * p.nThetaEff + l] *
                     p.cosmui_scaled[l * (p.mf + 1) + m];
    }
    p.amat[row * p.mnpd + mn] = accum;
}

// matrix = amat (flat copy).
template <class T>
__global__ void build_matrix_copy_kernel(LaplaceKernelParams<T> p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.mnpd * p.mnpd) return;
    p.matrix[i] = p.amat[i];
}

// Gauge rows: zero the (m=0, n in [-nf, 0)) rows across all columns (they
// duplicate the m=0/n>0 sin modes).
template <class T>
__global__ void build_matrix_gauge_kernel(LaplaceKernelParams<T> p) {
    const int mnp = blockIdx.x * blockDim.x + threadIdx.x;
    if (mnp >= p.mnpd) return;
    for (int all_n = 0; all_n < p.nf; ++all_n) {
        p.matrix[mnp * p.mnpd + all_n * (p.mf + 1)] = 0;
    }
}

// +0.5 diagonal (the 1/2*Phi jump term of the second-kind integral
// equation).
template <class T>
__global__ void build_matrix_diagonal_kernel(LaplaceKernelParams<T> p) {
    const int mn = blockIdx.x * blockDim.x + threadIdx.x;
    if (mn >= p.mnpd) return;
    p.matrix[mn * p.mnpd + mn] += T(0.5);
}

// bvec = bvec_sin + singular_bvec/nfp.
template <class T>
__global__ void assemble_bvec_kernel(LaplaceKernelParams<T> p) {
    const int mn = blockIdx.x * blockDim.x + threadIdx.x;
    if (mn >= p.mnpd) return;
    p.bvec[mn] = p.bvec_sin[mn] + p.singular_bvec[mn] * T(1.0 / p.nfp);
}

// Gauge RHS entries: zero (m=0, n in [-nf, 0)).
template <class T>
__global__ void gauge_bvec_kernel(LaplaceKernelParams<T> p) {
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

    upload_converted(cosnv_scaled, &cosnv_scaled_);
    upload_converted(sinnv_scaled, &sinnv_scaled_);
    upload_converted(cosmui_scaled, &cosmui_scaled_);
    upload_converted(sinmui_scaled, &sinmui_scaled_);

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
void LaplaceSolverOperator<T>::transform_greens_function_derivative(
    const T* d_greenp) {
    LaplaceKernelParams<T> p{};
    p.cosnv_scaled = cosnv_scaled_.data();
    p.sinnv_scaled = sinnv_scaled_.data();
    p.cosmui_scaled = cosmui_scaled_.data();
    p.sinmui_scaled = sinmui_scaled_.data();
    p.g1 = g1_.data();
    p.g2 = g2_.data();
    p.g1e = g1e_.data();
    p.g2e = g2e_.data();
    p.greenp = d_greenp;
    p.gstore = nullptr;
    p.singular_sin = nullptr;
    p.singular_cos = nullptr;
    p.singular_bvec = nullptr;
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
    p.grpmn_sin = grpmn_sin_.data();
    p.grpmn_cos = grpmn_cos_.data();
    p.gstore_symm = nullptr;
    p.bcos = nullptr;
    p.bsin = nullptr;
    p.actemp = nullptr;
    p.astemp = nullptr;
    p.bvec_sin = nullptr;
    p.amat = nullptr;
    p.matrix = nullptr;
    p.bvec = nullptr;

    launch_checked(
        reinterpret_cast<const void*>(&greenp_transform_phase_a_kernel<T>),
        sizes_.nZnT * sizes_.nThetaReduced, BLOCK_SIZE, p, stream_,
        "LaplaceSolverOperator: greenp transform phase A");
    launch_checked(
        reinterpret_cast<const void*>(&greenp_transform_phase_b_kernel<T>),
        sizes_.nZnT * (mf_ + 1), BLOCK_SIZE, p, stream_,
        "LaplaceSolverOperator: greenp transform phase B");
}

template <class T>
void LaplaceSolverOperator<T>::symmetrise_source_term(const T* d_gstore) {
    LaplaceKernelParams<T> p{};
    p.cosnv_scaled = cosnv_scaled_.data();
    p.sinnv_scaled = sinnv_scaled_.data();
    p.cosmui_scaled = cosmui_scaled_.data();
    p.sinmui_scaled = sinmui_scaled_.data();
    p.g1 = nullptr;
    p.g2 = nullptr;
    p.g1e = nullptr;
    p.g2e = nullptr;
    p.greenp = nullptr;
    p.gstore = d_gstore;
    p.singular_sin = nullptr;
    p.singular_cos = nullptr;
    p.singular_bvec = nullptr;
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
    p.grpmn_sin = nullptr;
    p.grpmn_cos = nullptr;
    p.gstore_symm = gstore_symm_.data();
    p.bcos = nullptr;
    p.bsin = nullptr;
    p.actemp = nullptr;
    p.astemp = nullptr;
    p.bvec_sin = nullptr;
    p.amat = nullptr;
    p.matrix = nullptr;
    p.bvec = nullptr;

    launch_checked(reinterpret_cast<const void*>(&symmetrise_gstore_kernel<T>),
                  sizes_.nThetaReduced * sizes_.nZeta, BLOCK_SIZE, p, stream_,
                  "LaplaceSolverOperator: symmetrise source");
}

template <class T>
void LaplaceSolverOperator<T>::accumulate_full_grpmn(const T* d_singular_sin,
                                                   const T* d_singular_cos) {
    LaplaceKernelParams<T> p{};
    p.cosnv_scaled = nullptr;
    p.sinnv_scaled = nullptr;
    p.cosmui_scaled = nullptr;
    p.sinmui_scaled = nullptr;
    p.g1 = nullptr;
    p.g2 = nullptr;
    p.g1e = nullptr;
    p.g2e = nullptr;
    p.greenp = nullptr;
    p.gstore = nullptr;
    p.singular_sin = d_singular_sin;
    p.singular_cos = d_singular_cos;
    p.singular_bvec = nullptr;
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
    p.grpmn_sin = grpmn_sin_.data();
    p.grpmn_cos = grpmn_cos_.data();
    p.gstore_symm = nullptr;
    p.bcos = nullptr;
    p.bsin = nullptr;
    p.actemp = nullptr;
    p.astemp = nullptr;
    p.bvec_sin = nullptr;
    p.amat = nullptr;
    p.matrix = nullptr;
    p.bvec = nullptr;

    launch_checked(reinterpret_cast<const void*>(&accumulate_full_grpmn_kernel<T>),
                  mnpd_ * sizes_.nZnT, BLOCK_SIZE, p, stream_,
                  "LaplaceSolverOperator: accumulate grpmn");
}

template <class T>
void LaplaceSolverOperator<T>::perform_toroidal_fourier_transforms() {
    LaplaceKernelParams<T> p{};
    p.cosnv_scaled = cosnv_scaled_.data();
    p.sinnv_scaled = sinnv_scaled_.data();
    p.cosmui_scaled = nullptr;
    p.sinmui_scaled = nullptr;
    p.g1 = nullptr;
    p.g2 = nullptr;
    p.g1e = nullptr;
    p.g2e = nullptr;
    p.greenp = nullptr;
    p.gstore = nullptr;
    p.singular_sin = nullptr;
    p.singular_cos = nullptr;
    p.singular_bvec = nullptr;
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
    p.grpmn_sin = grpmn_sin_.data();
    p.grpmn_cos = nullptr;
    p.gstore_symm = gstore_symm_.data();
    p.bcos = bcos_.data();
    p.bsin = bsin_.data();
    p.actemp = actemp_.data();
    p.astemp = astemp_.data();
    p.bvec_sin = nullptr;
    p.amat = nullptr;
    p.matrix = nullptr;
    p.bvec = nullptr;

    launch_checked(reinterpret_cast<const void*>(&toroidal_fft_phase_a_kernel<T>),
                  (nf_ + 1) * sizes_.nThetaReduced, BLOCK_SIZE, p, stream_,
                  "LaplaceSolverOperator: toroidal phase A");
    launch_checked(reinterpret_cast<const void*>(&toroidal_fft_phase_b_kernel<T>),
                  mnpd_ * (nf_ + 1) * sizes_.nThetaEff, BLOCK_SIZE, p, stream_,
                  "LaplaceSolverOperator: toroidal phase B");
}

template <class T>
void LaplaceSolverOperator<T>::perform_poloidal_fourier_transforms() {
    LaplaceKernelParams<T> p{};
    p.cosnv_scaled = nullptr;
    p.sinnv_scaled = nullptr;
    p.cosmui_scaled = cosmui_scaled_.data();
    p.sinmui_scaled = sinmui_scaled_.data();
    p.g1 = nullptr;
    p.g2 = nullptr;
    p.g1e = nullptr;
    p.g2e = nullptr;
    p.greenp = nullptr;
    p.gstore = nullptr;
    p.singular_sin = nullptr;
    p.singular_cos = nullptr;
    p.singular_bvec = nullptr;
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
    p.grpmn_sin = nullptr;
    p.grpmn_cos = nullptr;
    p.gstore_symm = nullptr;
    p.bcos = bcos_.data();
    p.bsin = bsin_.data();
    p.actemp = actemp_.data();
    p.astemp = astemp_.data();
    p.bvec_sin = bvec_sin_.data();
    p.amat = amat_.data();
    p.matrix = nullptr;
    p.bvec = nullptr;

    launch_checked(reinterpret_cast<const void*>(&poloidal_fft_bvec_kernel<T>),
                  mnpd_, BLOCK_SIZE, p, stream_,
                  "LaplaceSolverOperator: poloidal bvec");
    launch_checked(reinterpret_cast<const void*>(&poloidal_fft_amat_kernel<T>),
                  mnpd_ * mnpd_, BLOCK_SIZE, p, stream_,
                  "LaplaceSolverOperator: poloidal amat");
}

template <class T>
void LaplaceSolverOperator<T>::build_matrix() {
    LaplaceKernelParams<T> p{};
    p.cosnv_scaled = nullptr;
    p.sinnv_scaled = nullptr;
    p.cosmui_scaled = nullptr;
    p.sinmui_scaled = nullptr;
    p.g1 = nullptr;
    p.g2 = nullptr;
    p.g1e = nullptr;
    p.g2e = nullptr;
    p.greenp = nullptr;
    p.gstore = nullptr;
    p.singular_sin = nullptr;
    p.singular_cos = nullptr;
    p.singular_bvec = nullptr;
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
    p.grpmn_sin = nullptr;
    p.grpmn_cos = nullptr;
    p.gstore_symm = nullptr;
    p.bcos = nullptr;
    p.bsin = nullptr;
    p.actemp = nullptr;
    p.astemp = nullptr;
    p.bvec_sin = nullptr;
    p.amat = amat_.data();
    p.matrix = matrix_.data();
    p.bvec = nullptr;

    launch_checked(reinterpret_cast<const void*>(&build_matrix_copy_kernel<T>),
                  mnpd_ * mnpd_, BLOCK_SIZE, p, stream_,
                  "LaplaceSolverOperator: matrix copy");
    launch_checked(reinterpret_cast<const void*>(&build_matrix_gauge_kernel<T>),
                  mnpd_, BLOCK_SIZE, p, stream_,
                  "LaplaceSolverOperator: matrix gauge");
    launch_checked(reinterpret_cast<const void*>(&build_matrix_diagonal_kernel<T>),
                  mnpd_, BLOCK_SIZE, p, stream_,
                  "LaplaceSolverOperator: matrix diagonal");
}

template <class T>
void LaplaceSolverOperator<T>::decompose_matrix() {
    // The matrix is tiny: copy it to the host and factorize in double
    // (LuSolve, LAPACK dgetrf semantics on the same flat layout vmecpp
    // passed to LAPACK).
    matrix_.download(matrix_h_.data(), matrix_h_.size());
    const int info =
        LuSolve::decompose(matrix_h_.data(), pivots_h_.data(), mnpd_);
    if (info != 0) {
        throw VfieldError(
            "LaplaceSolverOperator::decompose_matrix: singular "
            "matrix (dgetrf-style info=" +
            std::to_string(info) + ")");
    }
    factorized_ = true;
}

template <class T>
void LaplaceSolverOperator<T>::solve_for_potential(const T* d_singular_bvec) {
    LaplaceKernelParams<T> p{};
    p.cosnv_scaled = nullptr;
    p.sinnv_scaled = nullptr;
    p.cosmui_scaled = nullptr;
    p.sinmui_scaled = nullptr;
    p.g1 = nullptr;
    p.g2 = nullptr;
    p.g1e = nullptr;
    p.g2e = nullptr;
    p.greenp = nullptr;
    p.gstore = nullptr;
    p.singular_sin = nullptr;
    p.singular_cos = nullptr;
    p.singular_bvec = d_singular_bvec;
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
    p.grpmn_sin = nullptr;
    p.grpmn_cos = nullptr;
    p.gstore_symm = nullptr;
    p.bcos = nullptr;
    p.bsin = nullptr;
    p.actemp = nullptr;
    p.astemp = nullptr;
    p.bvec_sin = bvec_sin_.data();
    p.amat = nullptr;
    p.matrix = nullptr;
    p.bvec = bvec_.data();

    launch_checked(reinterpret_cast<const void*>(&assemble_bvec_kernel<T>), mnpd_,
                  BLOCK_SIZE, p, stream_,
                  "LaplaceSolverOperator: assemble bvec");
    if (nf_ > 0) {
        launch_checked(reinterpret_cast<const void*>(&gauge_bvec_kernel<T>), nf_,
                      BLOCK_SIZE, p, stream_,
                      "LaplaceSolverOperator: gauge bvec");
    }

    // Solve on the host in double (the potential is control-adjacent; see
    // lu_solve.hpp).
    if (!factorized_) {
        throw VfieldError(
            "LaplaceSolverOperator::solve_for_potential: no factorization "
            "available — a partial update must follow a full update "
            "(decompose_matrix)");
    }
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
