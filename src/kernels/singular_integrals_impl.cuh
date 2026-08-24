// kernels/singular_integrals_impl.cuh — T/S recurrence and bvec/grpmn kernels.
//
// Included once per scalar type by singular_integrals_double.cu /
// singular_integrals_float.cu; see the explicit-instantiation split. The
// per-point recurrences run serial per thread (one thread per surface
// point); the bvec reduction is one thread per mode replaying vmecpp's
// fl -> row -> 4-wide-chunk accumulation order, so the results are
// summation-order-identical to the CPU reference.
#ifndef VFIELD_SRC_SINGULAR_INTEGRALS_IMPL_CUH_
#define VFIELD_SRC_SINGULAR_INTEGRALS_IMPL_CUH_

#include "vfield/free_boundary/singular_integrals_operator.hpp"

#include <cmath>
#include <numbers>
#include <type_traits>
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
struct SingularKernelParams {
    // Local quadratic forms of the metric (a, b2, c) and second fundamental
    // form (A, B2, C); passed explicitly so the unit tests can feed
    // synthetic geometry (vmecpp's prepareUpdate signature).
    const T* a;
    const T* b2;
    const T* c;
    const T* A;
    const T* B2;
    const T* C;
    // Fourier basis tables.
    const T* sinmu;
    const T* cosmu;
    const T* cosnv;
    const T* sinnv;
    const T* mscale;
    const T* nscale;
    // Source + weights + coefficients.
    const T* bDotN;
    const T* wInt;
    const T* cmns;
    // Sizes.
    int nf;
    int mf;
    int nZeta;
    int nThetaEff;
    int mnyq2;
    int nZnT;
    int mnfull;
    bool lasym;
    bool fullUpdate;
    // T/S tables and per-point constants.
    T* sqrtc2;
    T* sqrta2;
    T* r1p;
    T* r1m;
    T* r0p;
    T* r0m;
    T* ra1p;
    T* ra1m;
    T* tlp;
    T* tlm;
    T* slp;
    T* slm;
    // Outputs.
    T* bvecSin;
    T* bvecCos;
    T* grpmnSin;
    T* grpmnCos;
};

// Per-point constants and the T_l^+/- recurrences (and, on full updates, the
// S_l^+/- combinations). One thread per surface point; all recurrence state
// stays in registers.
template <class T>
__global__ void singularPrepareKernel(SingularKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZnT) return;

    const T a = p.a[kl];
    const T b2 = p.b2[kl];
    const T c = p.c[kl];

    // initialize constants (along expansion in l)
    const T ap = a + b2 + c;
    const T am = a - b2 + c;
    const T d = c - a;
    const T sqrtc2 = 2 * sqrt(c);
    const T sqrta2 = 2 * sqrt(a);

    p.sqrtc2[kl] = sqrtc2;
    p.sqrta2[kl] = sqrta2;

    T R1p = 0, R1m = 0, R0p = 0, R0m = 0, Ra1p = 0, Ra1m = 0;
    if (p.fullUpdate) {
        const T A = p.A[kl];
        const T B2 = p.B2[kl];
        const T C = p.C[kl];

        const T delta4 = ap * am - d * d;

        const T Ap = A + B2 + C;
        const T Am = A - B2 + C;
        const T D = C - A;

        R1p = (Ap * (delta4 - d * d) / ap - Am * ap + 2 * D * d) / delta4;
        R1m = (Am * (delta4 - d * d) / am - Ap * am + 2 * D * d) / delta4;
        R0p = (-Ap * am * d / ap - Am * d + 2 * D * am) / delta4;
        R0m = (-Am * ap * d / am - Ap * d + 2 * D * ap) / delta4;
        Ra1p = Ap / ap;
        Ra1m = Am / am;
    }

    p.r1p[kl] = R1p;
    p.r1m[kl] = R1m;
    p.r0p[kl] = R0p;
    p.r0m[kl] = R0m;
    p.ra1p[kl] = Ra1p;
    p.ra1m[kl] = Ra1m;

    const T sqrtap = sqrt(ap);
    const T sqrtam = sqrt(am);

    // Compute T^{\pm}_0 analytically (eq. 6.207 in
    // the_numerics_of_vmecpp.pdf).
    const T T0p =
        log((sqrtap * sqrtc2 + ap + d) / (sqrtap * sqrta2 - ap + d)) / sqrtap;
    const T T0m =
        log((sqrtam * sqrtc2 + am + d) / (sqrtam * sqrta2 - am + d)) / sqrtam;

    // Fill all Tlp[0..L] and Tlm[0..L] by picking the numerically stable
    // direction of the three-term recurrence on a per-(+/-), per-kl basis.
    //
    // The characteristic roots of the homogeneous recurrence satisfy
    //   A*r^2 + 2*d*r + B = 0  -> |r1 r2| = B/A.
    // For T^+: (A, B) = (ap, am), so |r1 r2| = am/ap.
    // For T^-: (A, B) = (am, ap), so |r1 r2| = ap/am.
    // If B > A (at least one |r| > 1), forward iteration is unstable and
    // backward (Miller's algorithm) is used instead; otherwise forward is
    // fine.
    //
    // T^{\pm}_0 is analytic (above); T^{\pm}_{-1} = 0. Forward produces
    // T_{l+1} from T_l and T_{l-1}; backward produces T_{l-1} from T_l and
    // T_{l+1} via the same recurrence solved in reverse. For backward,
    // iteration starts from a zero seed far above the required L; the result
    // is then normalized to match the analytic T^{\pm}_0.
    //
    // rhs(l+1) = sqrtc2 + (-1)^{l+1}*sqrta2  (same for T^+ and T^-).
    const int L = p.mf + p.nf;
    // The spurious solution is damped by (A/B)^TAIL_EXTRA per pass.
    // For the worst realistic ratio (A/B ~ 0.5) suppression is ~0.5^50 ~
    // 1e-16.
    const int TAIL_EXTRA = 50;
    const int L_TAIL = L + TAIL_EXTRA;

    // Only switch to backward when the forward spurious-mode growth
    // (|r1 r2| = B/A) would actually exceed double precision over L steps.
    // Threshold: forward is considered stable as long as (B/A)^L < 1e10,
    // i.e. spurious amplitude stays within ~1e10 of the particular solution.
    // Near-degenerate kl (|r1|~|r2|~1) fall in the forward branch, where
    // zero-seed Miller is known to misconverge (spurious modes never damp).
    // Formula: L * ln(B/A) < ln(1e10) -> B/A < exp(ln(1e10)/L).
    constexpr double LOG_GROWTH_THRESHOLD = 10.0 * 2.30258509299;  // ln(1e10)
    const T logRatioP = (am > ap && ap > T(0)) ? log(am / ap) : T(0);
    const bool useBackwardP =
        static_cast<double>(L) * static_cast<double>(logRatioP) >
        LOG_GROWTH_THRESHOLD;
    const T logRatioM = (ap > am && am > T(0)) ? log(ap / am) : T(0);
    const bool useBackwardM =
        static_cast<double>(L) * static_cast<double>(logRatioM) >
        LOG_GROWTH_THRESHOLD;

    // Miller seed: double keeps vmecpp's 1e-300; float needs a value above
    // the denormal range (1e-300 underflows float entirely).
    const T seed = []() -> T {
        if constexpr (std::is_same_v<T, double>) {
            return T(1.0e-300);
        } else {
            return T(1.0e-30);
        }
    }();

    // --- T^+: A = ap, B = am ---
    p.tlp[kl] = T0p;
    if (useBackwardP) {
        // forward unstable -> use backward recurrence.
        T T_hi = 0;
        T T_cur = seed;
        for (int l = L_TAIL; l >= 1; --l) {
            const T rhs = sqrtc2 + (l % 2 == 0 ? T(-1) : T(1)) * sqrta2;
            const T T_lo =
                (rhs - T(2 * l + 1) * d * T_cur - T(l + 1) * ap * T_hi) /
                (T(l) * am);
            T_hi = T_cur;
            T_cur = T_lo;
            if (l - 1 <= L) { p.tlp[(l - 1) * p.nZnT + kl] = T_lo; }
        }
        const T scaleP = T0p / p.tlp[kl];
        for (int l = 0; l <= L; ++l) { p.tlp[l * p.nZnT + kl] *= scaleP; }
    } else {
        // forward stable.
        T T_prev = 0;  // T^+_{-1}
        int sgn = 1;
        for (int fl = 0; fl < L; ++fl) {
            sgn = -sgn;
            const T rhs = sqrtc2 + T(sgn) * sqrta2;
            const T T_next =
                (rhs - T(2 * fl + 1) * d * p.tlp[fl * p.nZnT + kl] -
                 T(fl) * am * T_prev) /
                (ap * T(fl + 1));
            T_prev = p.tlp[fl * p.nZnT + kl];
            p.tlp[(fl + 1) * p.nZnT + kl] = T_next;
        }
    }

    // --- T^-: A = am, B = ap ---
    p.tlm[kl] = T0m;
    if (useBackwardM) {
        // forward unstable -> use backward recurrence.
        T T_hi = 0;
        T T_cur = seed;
        for (int l = L_TAIL; l >= 1; --l) {
            const T rhs = sqrtc2 + (l % 2 == 0 ? T(-1) : T(1)) * sqrta2;
            const T T_lo =
                (rhs - T(2 * l + 1) * d * T_cur - T(l + 1) * am * T_hi) /
                (T(l) * ap);
            T_hi = T_cur;
            T_cur = T_lo;
            if (l - 1 <= L) { p.tlm[(l - 1) * p.nZnT + kl] = T_lo; }
        }
        const T scaleM = T0m / p.tlm[kl];
        for (int l = 0; l <= L; ++l) { p.tlm[l * p.nZnT + kl] *= scaleM; }
    } else {
        // forward stable.
        T T_prev = 0;  // T^-_{-1}
        int sgn = 1;
        for (int fl = 0; fl < L; ++fl) {
            sgn = -sgn;
            const T rhs = sqrtc2 + T(sgn) * sqrta2;
            const T T_next =
                (rhs - T(2 * fl + 1) * d * p.tlm[fl * p.nZnT + kl] -
                 T(fl) * ap * T_prev) /
                (am * T(fl + 1));
            T_prev = p.tlm[fl * p.nZnT + kl];
            p.tlm[(fl + 1) * p.nZnT + kl] = T_next;
        }
    }

    // --- S_l^+/- from T (Eq. (A17)) ---
    if (p.fullUpdate) {
        T Tl1p = 0;  // T^+_{-1}
        T Tl1m = 0;  // T^-_{-1}
        int sgn = 1;
        for (int fl = 0; fl < 1 + p.nf + p.mf; ++fl) {
            p.slp[fl * p.nZnT + kl] =
                (R1p * T(fl) + Ra1p) * p.tlp[fl * p.nZnT + kl] +
                R0p * T(fl) * Tl1p - (R0p + R1p) / sqrtc2 +
                T(sgn) * (R0p - R1p) / sqrta2;
            p.slm[fl * p.nZnT + kl] =
                (R1m * T(fl) + Ra1m) * p.tlm[fl * p.nZnT + kl] +
                R0m * T(fl) * Tl1m - (R0m + R1m) / sqrtc2 +
                T(sgn) * (R0m - R1m) / sqrta2;

            sgn = -sgn;
            Tl1p = p.tlp[fl * p.nZnT + kl];
            Tl1m = p.tlm[fl * p.nZnT + kl];
        }
    }
}

// Singular part of the Laplace RHS: one thread per mode, replaying vmecpp's
// fl -> row -> 4-wide-chunk accumulation order exactly (the buf[] pattern is
// a CPU-vectorization artifact but defines the summation order).
template <class T>
__global__ void singularBvecKernel(SingularKernelParams<T> p) {
    const int mn = blockIdx.x * blockDim.x + threadIdx.x;
    if (mn >= p.mnfull) return;

    const int n = mn / (p.mf + 1) - p.nf;  // -nf:nf
    const int m = mn % (p.mf + 1);
    const int n_abs = abs(n);
    const bool is_posn = (n >= 0);

    T accum = 0;
    // The 4-wide chunk accumulators (per row), summed after the row loop.
    T buf[4] = {0, 0, 0, 0};

    for (int fl = 0; fl < 1 + p.nf + p.mf; ++fl) {
        const T cmns_factor =
            p.cmns[(fl * (p.nf + 1) + n_abs) * (p.mf + 1) + m] /
            (p.mscale[m] * p.nscale[n_abs]);

        const T* tl = (is_posn) ? &p.tlp[fl * p.nZnT] : &p.tlm[fl * p.nZnT];

        for (int l = 0; l < p.nThetaEff; ++l) {
            const int idx_lm = l * (p.mnyq2 + 1) + m;
            const T sinmu_l = p.sinmu[idx_lm];
            const T cosmu_l = p.cosmu[idx_lm];

            if (m == 0 || n_abs == 0) {
                // analysum: only the +n mode is accumulated, with
                // (Tlp + Tlm); the -n threads do nothing.
                if (is_posn) {
                    for (int k = 0; k < p.nZeta; ++k) {
                        const int klRel = l * p.nZeta + k;
                        const int idx_nk = n_abs * p.nZeta + k;
                        // sin(mu - |n|v) * cmns(l,n,m)
                        const T sinp = (sinmu_l * p.cosnv[idx_nk] -
                                        cosmu_l * p.sinnv[idx_nk]) *
                                       cmns_factor;
                        accum += (p.tlp[fl * p.nZnT + klRel] +
                                  p.tlm[fl * p.nZnT + klRel]) *
                                 p.bDotN[klRel] * p.wInt[l] * sinp;
                    }
                }
            } else {
                // analysum2: +n and -n modes accumulate separately, with the
                // 4-wide chunk pattern of the vmecpp loop.
                const T coeff1 = sinmu_l * cmns_factor;
                const T coeff2 = cosmu_l * cmns_factor;

                buf[0] = 0;
                buf[1] = 0;
                buf[2] = 0;
                buf[3] = 0;
                int k = 0;
                for (; k + 3 < p.nZeta; k += 4) {
                    const int klRel = l * p.nZeta + k;
                    const int idx_nk = n_abs * p.nZeta + k;
                    T c0 = p.bDotN[klRel + 0] * p.wInt[l];
                    T c1 = p.bDotN[klRel + 1] * p.wInt[l];
                    T c2 = p.bDotN[klRel + 2] * p.wInt[l];
                    T c3 = p.bDotN[klRel + 3] * p.wInt[l];

                    T factor0, factor1, factor2, factor3;
                    if (is_posn) {
                        // sin(mu - |n|v) * cmns(l,n,m)
                        factor0 = coeff1 * p.cosnv[idx_nk + 0] -
                                  coeff2 * p.sinnv[idx_nk + 0];
                        factor1 = coeff1 * p.cosnv[idx_nk + 1] -
                                  coeff2 * p.sinnv[idx_nk + 1];
                        factor2 = coeff1 * p.cosnv[idx_nk + 2] -
                                  coeff2 * p.sinnv[idx_nk + 2];
                        factor3 = coeff1 * p.cosnv[idx_nk + 3] -
                                  coeff2 * p.sinnv[idx_nk + 3];
                    } else {
                        // sin(mu + |n|v) * cmns(l,n,m)
                        factor0 = coeff1 * p.cosnv[idx_nk + 0] +
                                  coeff2 * p.sinnv[idx_nk + 0];
                        factor1 = coeff1 * p.cosnv[idx_nk + 1] +
                                  coeff2 * p.sinnv[idx_nk + 1];
                        factor2 = coeff1 * p.cosnv[idx_nk + 2] +
                                  coeff2 * p.sinnv[idx_nk + 2];
                        factor3 = coeff1 * p.cosnv[idx_nk + 3] +
                                  coeff2 * p.sinnv[idx_nk + 3];
                    }

                    buf[0] += tl[klRel + 0] * c0 * factor0;
                    buf[1] += tl[klRel + 1] * c1 * factor1;
                    buf[2] += tl[klRel + 2] * c2 * factor2;
                    buf[3] += tl[klRel + 3] * c3 * factor3;
                }
                accum += buf[0] + buf[1] + buf[2] + buf[3];
                for (; k < p.nZeta; ++k) {
                    const int klRel = l * p.nZeta + k;
                    const int idx_nk = n_abs * p.nZeta + k;
                    const T coeff1k = sinmu_l * p.cosnv[idx_nk] * cmns_factor;
                    const T coeff2k = cosmu_l * p.sinnv[idx_nk] * cmns_factor;

                    T factor;
                    if (is_posn) {
                        // sin(mu - |n|v) * cmns(l,n,m)
                        factor = coeff1k - coeff2k;
                    } else {
                        // sin(mu + |n|v) * cmns(l,n,m)
                        factor = coeff1k + coeff2k;
                    }
                    const T c = p.bDotN[klRel] * p.wInt[l];
                    accum += tl[klRel] * c * factor;
                }
            }
        }
    }

    p.bvecSin[mn] = accum;
    if (p.lasym) {
        // TODO(port): the cos-part needs the cos(mu -+ |n|v) factors; the
        // symmetric golden case does not exercise it. Implemented when the
        // lasym golden data becomes available.
        p.bvecCos[mn] = 0;
    }
}

// Singular part of the system-matrix kernel: one thread per (mode, point),
// accumulating over fl in vmecpp's order.
template <class T>
__global__ void singularGrpmnKernel(SingularKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int mn = idx / p.nZnT;
    const int klRel = idx - mn * p.nZnT;
    if (mn >= p.mnfull) return;

    const int n = mn / (p.mf + 1) - p.nf;  // -nf:nf
    const int m = mn % (p.mf + 1);
    const int n_abs = abs(n);
    const bool is_posn = (n >= 0);

    const int l = klRel / p.nZeta;
    const int k = klRel % p.nZeta;
    const int idx_lm = l * (p.mnyq2 + 1) + m;
    const int idx_nk = n_abs * p.nZeta + k;

    const T sinmu_l = p.sinmu[idx_lm];
    const T cosmu_l = p.cosmu[idx_lm];

    T accum = 0;
    for (int fl = 0; fl < 1 + p.nf + p.mf; ++fl) {
        const T cmns_factor =
            p.cmns[(fl * (p.nf + 1) + n_abs) * (p.mf + 1) + m] /
            (p.mscale[m] * p.nscale[n_abs]);

        const T sl =
            (is_posn) ? p.slp[fl * p.nZnT + klRel] : p.slm[fl * p.nZnT + klRel];

        if (m == 0 || n_abs == 0) {
            // sin(mu - |n|v) * cmns(l,n,m); only the +n mode.
            if (is_posn) {
                const T sinp =
                    (sinmu_l * p.cosnv[idx_nk] - cosmu_l * p.sinnv[idx_nk]) *
                    cmns_factor;
                accum +=
                    (p.slp[fl * p.nZnT + klRel] + p.slm[fl * p.nZnT + klRel]) *
                    sinp;
            }
        } else {
            T factor;
            if (is_posn) {
                // sin(mu - |n|v) * cmns(l,n,m)
                factor =
                    (sinmu_l * p.cosnv[idx_nk] - cosmu_l * p.sinnv[idx_nk]) *
                    cmns_factor;
            } else {
                // sin(mu + |n|v) * cmns(l,n,m)
                factor =
                    (sinmu_l * p.cosnv[idx_nk] + cosmu_l * p.sinnv[idx_nk]) *
                    cmns_factor;
            }
            accum += sl * factor;
        }
    }

    p.grpmnSin[mn * p.nZnT + klRel] = accum;
    if (p.lasym) { p.grpmnCos[mn * p.nZnT + klRel] = 0; }
}

template <class T>
SingularIntegralsOperator<T>::SingularIntegralsOperator(
    const Sizes& sizes,
    const FourierBasisDevice<T>& fb,
    const SurfaceGeometryOperator<T>& sg)
    : sizes_(sizes), fb_(fb), sg_(sg), stream_(nullptr) {
    nf_ = sizes_.ntor;
    mf_ = sizes_.mpol + 1;
    mnfull_ = (2 * nf_ + 1) * (mf_ + 1);

    SingularCoefficients coeffs(nf_, mf_);
    uploadConverted(coeffs.cmns, &cmns_);
    uploadConverted(sizes_.wInt, &w_int_);

    sqrtc2_.allocate(sizes_.nZnT);
    sqrta2_.allocate(sizes_.nZnT);
    r1p_.allocate(sizes_.nZnT);
    r1m_.allocate(sizes_.nZnT);
    r0p_.allocate(sizes_.nZnT);
    r0m_.allocate(sizes_.nZnT);
    ra1p_.allocate(sizes_.nZnT);
    ra1m_.allocate(sizes_.nZnT);
    tlp_.allocate((nf_ + mf_ + 2) * sizes_.nZnT);
    tlm_.allocate((nf_ + mf_ + 2) * sizes_.nZnT);
    slp_.allocate((nf_ + mf_ + 1) * sizes_.nZnT);
    slm_.allocate((nf_ + mf_ + 1) * sizes_.nZnT);

    bvec_sin_.allocate(mnfull_);
    grpmn_sin_.allocate(mnfull_ * sizes_.nZnT);
    if (sizes_.lasym) {
        bvec_cos_.allocate(mnfull_);
        grpmn_cos_.allocate(mnfull_ * sizes_.nZnT);
    }
}

template <class T>
void SingularIntegralsOperator<T>::update(const T* d_bdotn, bool full_update) {
    prepareUpdate(sg_.guu(), sg_.guv(), sg_.gvv(), sg_.auu(), sg_.auv(),
                  sg_.avv(), full_update);
    performUpdate(d_bdotn, full_update);
}

template <class T>
void SingularIntegralsOperator<T>::prepareUpdate(const T* d_a,
                                                 const T* d_b2,
                                                 const T* d_c,
                                                 const T* d_A,
                                                 const T* d_B2,
                                                 const T* d_C,
                                                 bool full_update) {
    SingularKernelParams<T> p{};
    p.a = d_a;
    p.b2 = d_b2;
    p.c = d_c;
    p.A = d_A;
    p.B2 = d_B2;
    p.C = d_C;
    p.sinmu = fb_.sinmu();
    p.cosmu = fb_.cosmu();
    p.cosnv = fb_.cosnv();
    p.sinnv = fb_.sinnv();
    p.mscale = fb_.mscale();
    p.nscale = fb_.nscale();
    p.bDotN = nullptr;
    p.wInt = w_int_.data();
    p.cmns = cmns_.data();
    p.nf = nf_;
    p.mf = mf_;
    p.nZeta = sizes_.nZeta;
    p.nThetaEff = sizes_.nThetaEff;
    p.mnyq2 = sizes_.mnyq2;
    p.nZnT = sizes_.nZnT;
    p.mnfull = mnfull_;
    p.lasym = sizes_.lasym;
    p.fullUpdate = full_update;
    p.sqrtc2 = sqrtc2_.data();
    p.sqrta2 = sqrta2_.data();
    p.r1p = r1p_.data();
    p.r1m = r1m_.data();
    p.r0p = r0p_.data();
    p.r0m = r0m_.data();
    p.ra1p = ra1p_.data();
    p.ra1m = ra1m_.data();
    p.tlp = tlp_.data();
    p.tlm = tlm_.data();
    p.slp = slp_.data();
    p.slm = slm_.data();
    p.bvecSin = bvec_sin_.data();
    p.bvecCos = bvec_cos_.data();
    p.grpmnSin = grpmn_sin_.data();
    p.grpmnCos = grpmn_cos_.data();

    launchChecked(reinterpret_cast<const void*>(&singularPrepareKernel<T>),
                  sizes_.nZnT, BLOCK_SIZE, p, stream_,
                  "SingularIntegralsOperator::prepareUpdate: prepare");
}

template <class T>
void SingularIntegralsOperator<T>::performUpdate(const T* d_bdotn,
                                                 bool full_update) {
    SingularKernelParams<T> p{};
    p.a = nullptr;
    p.b2 = nullptr;
    p.c = nullptr;
    p.A = nullptr;
    p.B2 = nullptr;
    p.C = nullptr;
    p.sinmu = fb_.sinmu();
    p.cosmu = fb_.cosmu();
    p.cosnv = fb_.cosnv();
    p.sinnv = fb_.sinnv();
    p.mscale = fb_.mscale();
    p.nscale = fb_.nscale();
    p.bDotN = d_bdotn;
    p.wInt = w_int_.data();
    p.cmns = cmns_.data();
    p.nf = nf_;
    p.mf = mf_;
    p.nZeta = sizes_.nZeta;
    p.nThetaEff = sizes_.nThetaEff;
    p.mnyq2 = sizes_.mnyq2;
    p.nZnT = sizes_.nZnT;
    p.mnfull = mnfull_;
    p.lasym = sizes_.lasym;
    p.fullUpdate = full_update;
    p.sqrtc2 = sqrtc2_.data();
    p.sqrta2 = sqrta2_.data();
    p.r1p = r1p_.data();
    p.r1m = r1m_.data();
    p.r0p = r0p_.data();
    p.r0m = r0m_.data();
    p.ra1p = ra1p_.data();
    p.ra1m = ra1m_.data();
    p.tlp = tlp_.data();
    p.tlm = tlm_.data();
    p.slp = slp_.data();
    p.slm = slm_.data();
    p.bvecSin = bvec_sin_.data();
    p.bvecCos = bvec_cos_.data();
    p.grpmnSin = grpmn_sin_.data();
    p.grpmnCos = grpmn_cos_.data();

    launchChecked(reinterpret_cast<const void*>(&singularBvecKernel<T>),
                  mnfull_, BLOCK_SIZE, p, stream_,
                  "SingularIntegralsOperator::performUpdate: bvec");
    if (full_update) {
        launchChecked(reinterpret_cast<const void*>(&singularGrpmnKernel<T>),
                      mnfull_ * sizes_.nZnT, BLOCK_SIZE, p, stream_,
                      "SingularIntegralsOperator::performUpdate: grpmn");
    }
}

}  // namespace vfield

#endif  // VFIELD_SRC_SINGULAR_INTEGRALS_IMPL_CUH_
