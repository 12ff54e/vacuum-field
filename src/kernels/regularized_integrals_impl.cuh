// kernels/regularized_integrals_impl.cuh — regularized Green's-function kernels.
//
// Included once per scalar type by regularized_integrals_double.cu /
// regularized_integrals_float.cu; see the explicit-instantiation split. Each
// output element is written by exactly one thread: greenp per (source,
// target) pair accumulating over the field periods in ascending order, and
// gstore per target point accumulating over the source points in ascending
// order — vmecpp's exact summation order.
#ifndef VFIELD_SRC_REGULARIZED_INTEGRALS_IMPL_CUH_
#define VFIELD_SRC_REGULARIZED_INTEGRALS_IMPL_CUH_

#include "vfield/free_boundary/regularized_integrals_operator.hpp"

#include <cmath>
#include <numbers>
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
struct RegularizedKernelParams {
    // Surface geometry.
    const T* rzb2;
    const T* z1b;
    const T* drv;
    const T* snz;
    const T* snr;
    const T* snv;
    const T* rcosuv;
    const T* rsinuv;
    const T* r1b;
    const T* guu;
    const T* guv;
    const T* gvv;
    const T* auu;
    const T* auv;
    const T* avv;
    // Field-period rotation tables (from the surface geometry).
    const T* cosPer;
    const T* sinPer;
    // Tangential angle tables + weights.
    const T* tanu;
    const T* tanv;
    const T* tanvPer;
    const T* bDotN;
    const T* wInt;
    // Sizes.
    int nZeta;
    int nThetaEven;
    int nZnT;
    int nfp;
    int nvper;
    bool axisymmetric;
    // Outputs.
    T* greenp;
    T* gstore;
};

// 3D case: regularized normal-derivative kernel. One thread per (source,
// target) pair, accumulating over the field periods in ascending order.
template <class T>
__global__ void regularizedGreenpKernel(RegularizedKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int full = p.nThetaEven * p.nZeta;
    const int klp = idx / full;
    const int kl = idx - klp * full;
    if (klp >= p.nZnT) return;

    const int lp = klp / p.nZeta;
    const int kp = klp % p.nZeta;
    const int l = kl / p.nZeta;
    const int k = kl % p.nZeta;

    const T twopidivnfp = T(2.0 * std::numbers::pi) / T(p.nfp);

    const T xp = p.rcosuv[klp];
    const T yp = p.rsinuv[klp];

    const T gsave = p.rzb2[klp] + p.rzb2[kl] - 2 * p.z1b[kl] * p.z1b[klp];
    const T dsave = p.drv[klp] + p.z1b[kl] * p.snz[klp];

    const int delta_l = (l - lp + p.nThetaEven) % p.nThetaEven;
    const int delta_k = (k - kp + p.nZeta) % p.nZeta;

    T accum = 0;

    // first field period (exact singularity skipped): subtract the analytic
    // approximation.
    if (kl != klp) {
        // xper == xp, yper == yp in the first period
        const T sxsave = (p.snr[klp] * xp - p.snv[klp] * yp) / p.r1b[klp];
        const T sysave = (p.snr[klp] * yp + p.snv[klp] * xp) / p.r1b[klp];

        T ga1 = p.guu[klp] * p.tanu[delta_l] * p.tanu[delta_l] +
                p.guv[klp] * p.tanu[delta_l] * p.tanv[delta_k] +
                p.gvv[klp] * p.tanv[delta_k] * p.tanv[delta_k];
        T ga2 = p.auu[klp] * p.tanu[delta_l] * p.tanu[delta_l] +
                p.auv[klp] * p.tanu[delta_l] * p.tanv[delta_k] +
                p.avv[klp] * p.tanv[delta_k] * p.tanv[delta_k];
        ga2 /= ga1;
        ga1 = 1 / sqrt(ga1);

        const T ftemp =
            1 / (gsave - 2 * (xp * p.rcosuv[kl] + yp * p.rsinuv[kl]));
        const T htemp = sqrt(ftemp);

        // 2 pi from Laplace equation
        // 1/nfp to make the toroidal integral below over the whole machine
        accum += twopidivnfp *
                 (htemp * ftemp *
                      (p.rcosuv[kl] * sxsave + p.rsinuv[kl] * sysave + dsave) -
                  ga1 * ga2);
    }

    // all following field periods: rotated evaluation point, no analytic
    // subtraction.
    for (int per = 1; per < p.nfp; ++per) {
        const T cos_per = p.cosPer[per];
        const T sin_per = p.sinPer[per];
        const T xper = xp * cos_per - yp * sin_per;
        const T yper = xp * sin_per + yp * cos_per;
        const T sxsave = (p.snr[klp] * xper - p.snv[klp] * yper) / p.r1b[klp];
        const T sysave = (p.snr[klp] * yper + p.snv[klp] * xper) / p.r1b[klp];

        const T ftemp =
            1 / (gsave - 2 * (xper * p.rcosuv[kl] + yper * p.rsinuv[kl]));
        const T htemp = sqrt(ftemp);

        accum += twopidivnfp * htemp * ftemp *
                 (p.rcosuv[kl] * sxsave + p.rsinuv[kl] * sysave + dsave);
    }  // per

    p.greenp[klp * full + kl] = accum;
}

// 3D case: regularized source term. One thread per target point,
// accumulating over the source points in ascending order.
template <class T>
__global__ void regularizedGstoreKernel(RegularizedKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    const int full = p.nThetaEven * p.nZeta;
    if (kl >= full) return;

    const int l = kl / p.nZeta;
    const T twopidivnfp = T(2.0 * std::numbers::pi) / T(p.nfp);

    T accum = 0;
    for (int klp = 0; klp < p.nZnT; ++klp) {
        const int lp = klp / p.nZeta;
        const int kp = klp % p.nZeta;
        const T bexni = p.bDotN[klp] * p.wInt[lp];

        const T xp = p.rcosuv[klp];
        const T yp = p.rsinuv[klp];

        const T gsave = p.rzb2[klp] + p.rzb2[kl] - 2 * p.z1b[kl] * p.z1b[klp];

        const int delta_l = (l - lp + p.nThetaEven) % p.nThetaEven;

        // first field period (exact singularity skipped).
        if (kl != klp) {
            const int delta_k = (kl % p.nZeta - kp + p.nZeta) % p.nZeta;
            T ga1 = p.guu[klp] * p.tanu[delta_l] * p.tanu[delta_l] +
                    p.guv[klp] * p.tanu[delta_l] * p.tanv[delta_k] +
                    p.gvv[klp] * p.tanv[delta_k] * p.tanv[delta_k];
            ga1 = 1 / sqrt(ga1);

            const T ftemp =
                1 / (gsave - 2 * (xp * p.rcosuv[kl] + yp * p.rsinuv[kl]));
            const T htemp = sqrt(ftemp);

            accum += bexni * twopidivnfp * (htemp - ga1);
        }

        // all following field periods.
        for (int per = 1; per < p.nfp; ++per) {
            const T cos_per = p.cosPer[per];
            const T sin_per = p.sinPer[per];
            const T xper = xp * cos_per - yp * sin_per;
            const T yper = xp * sin_per + yp * cos_per;

            const T ftemp =
                1 / (gsave - 2 * (xper * p.rcosuv[kl] + yper * p.rsinuv[kl]));
            const T htemp = sqrt(ftemp);

            accum += bexni * twopidivnfp * htemp;
        }  // per
    }  // klp

    p.gstore[kl] = accum;
}

// Axisymmetric case (nZeta == 1): the toroidal integral runs over nvper
// toroidal images of the evaluation point; the analytic approximation is
// subtracted at every image.
template <class T>
__global__ void regularizedGreenpAxisymKernel(RegularizedKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int klp = idx / p.nThetaEven;
    const int kl = idx - klp * p.nThetaEven;
    if (klp >= p.nZnT) return;

    const T measure = T(2.0 * std::numbers::pi) / T(p.nvper);

    const T xp = p.rcosuv[klp];
    const T yp = p.rsinuv[klp];

    const T gsave = p.rzb2[klp] + p.rzb2[kl] - 2 * p.z1b[kl] * p.z1b[klp];
    const T dsave = p.drv[klp] + p.z1b[kl] * p.snz[klp];

    const int delta_l = (kl - klp + p.nThetaEven) % p.nThetaEven;

    T accum = 0;
    for (int per = 0; per < p.nvper; ++per) {
        // The exact singularity (image coincides with the source point) is
        // handled analytically in SingularIntegrals; skip it here.
        if (per == 0 && kl == klp) { continue; }

        const T cosper = cos(measure * per);
        const T sinper = sin(measure * per);

        const T xper = xp * cosper - yp * sinper;
        const T yper = xp * sinper + yp * cosper;

        const T sxsave = (p.snr[klp] * xper - p.snv[klp] * yper) / p.r1b[klp];
        const T sysave = (p.snr[klp] * yper + p.snv[klp] * xper) / p.r1b[klp];

        const T tanv_p = p.tanvPer[per];

        T ga1 = p.guu[klp] * p.tanu[delta_l] * p.tanu[delta_l] +
                p.guv[klp] * p.tanu[delta_l] * tanv_p +
                p.gvv[klp] * tanv_p * tanv_p;
        T ga2 = p.auu[klp] * p.tanu[delta_l] * p.tanu[delta_l] +
                p.auv[klp] * p.tanu[delta_l] * tanv_p +
                p.avv[klp] * tanv_p * tanv_p;
        ga2 /= ga1;
        ga1 = 1 / sqrt(ga1);

        const T ftemp =
            1 / (gsave - 2 * (xper * p.rcosuv[kl] + yper * p.rsinuv[kl]));
        const T htemp = sqrt(ftemp);

        accum += measure *
                 (htemp * ftemp *
                      (p.rcosuv[kl] * sxsave + p.rsinuv[kl] * sysave + dsave) -
                  ga1 * ga2);
    }  // per

    p.greenp[klp * p.nThetaEven + kl] = accum;
}

// Axisymmetric case: regularized source term.
template <class T>
__global__ void regularizedGstoreAxisymKernel(RegularizedKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nThetaEven) return;

    const T measure = T(2.0 * std::numbers::pi) / T(p.nvper);

    T accum = 0;
    for (int klp = 0; klp < p.nZnT; ++klp) {
        const T bexni = p.bDotN[klp] * p.wInt[klp];

        const T xp = p.rcosuv[klp];
        const T yp = p.rsinuv[klp];

        const T gsave = p.rzb2[klp] + p.rzb2[kl] - 2 * p.z1b[kl] * p.z1b[klp];

        const int delta_l = (kl - klp + p.nThetaEven) % p.nThetaEven;

        for (int per = 0; per < p.nvper; ++per) {
            if (per == 0 && kl == klp) { continue; }

            const T cosper = cos(measure * per);
            const T sinper = sin(measure * per);

            const T xper = xp * cosper - yp * sinper;
            const T yper = xp * sinper + yp * cosper;

            const T tanv_p = p.tanvPer[per];

            T ga1 = p.guu[klp] * p.tanu[delta_l] * p.tanu[delta_l] +
                    p.guv[klp] * p.tanu[delta_l] * tanv_p +
                    p.gvv[klp] * tanv_p * tanv_p;
            ga1 = 1 / sqrt(ga1);

            const T ftemp =
                1 / (gsave - 2 * (xper * p.rcosuv[kl] + yper * p.rsinuv[kl]));
            const T htemp = sqrt(ftemp);

            accum += bexni * measure * (htemp - ga1);
        }  // per
    }  // klp

    p.gstore[kl] = accum;
}

template <class T>
RegularizedIntegralsOperator<T>::RegularizedIntegralsOperator(
    const Sizes& sizes,
    const SurfaceGeometryOperator<T>& sg)
    : sizes_(sizes), sg_(sg), stream_(nullptr) {
    nvper_ = (sizes_.nZeta == 1) ? 64 : sizes_.nfp;

    // tan-half-angle tables (host-computed, vmecpp's computeConstants).
    const double epsTan = 1.0e-15;
    const double bigNo = 1.0e50;

    std::vector<double> tanu(sizes_.nThetaEven);
    for (int l = 0; l < sizes_.nThetaEven; ++l) {
        const double argu = std::numbers::pi / sizes_.nThetaEven * l;
        if (std::abs(argu - 0.5 * std::numbers::pi) < epsTan) {
            // mask singularities at pi/2
            tanu[l] = bigNo;
        } else {
            tanu[l] = 2.0 * std::tan(argu);
        }
    }  // l

    std::vector<double> tanv(sizes_.nZeta);
    for (int k = 0; k < sizes_.nZeta; ++k) {
        const double argv = std::numbers::pi / sizes_.nZeta * k;
        if (std::abs(argv - 0.5 * std::numbers::pi) < epsTan) {
            // mask singularity at pi/2
            tanv[k] = bigNo;
        } else {
            tanv[k] = 2.0 * std::tan(argv);
        }
    }  // k

    std::vector<double> tanv_per;
    if (sizes_.nZeta == 1) {
        tanv_per.resize(nvper_);
        for (int per = 0; per < nvper_; ++per) {
            const double argv = std::numbers::pi * per / nvper_;
            if (std::abs(argv - 0.5 * std::numbers::pi) < epsTan) {
                tanv_per[per] = bigNo;
            } else {
                tanv_per[per] = 2.0 * std::tan(argv);
            }
        }  // per
    }

    uploadConverted(tanu, &tanu_);
    uploadConverted(tanv, &tanv_);
    uploadConverted(tanv_per, &tanv_per_);
    uploadConverted(sizes_.wInt, &w_int_);

    greenp_.allocate(sizes_.nZnT * sizes_.nThetaEven * sizes_.nZeta);
    gstore_.allocate(sizes_.nThetaEven * sizes_.nZeta);
}

template <class T>
void RegularizedIntegralsOperator<T>::update(const T* d_bdotn) {
    RegularizedKernelParams<T> p{};
    p.rzb2 = sg_.rzb2();
    p.z1b = sg_.z1b();
    p.drv = sg_.drv();
    p.snz = sg_.snz();
    p.snr = sg_.snr();
    p.snv = sg_.snv();
    p.rcosuv = sg_.rcosuv();
    p.rsinuv = sg_.rsinuv();
    p.r1b = sg_.r1b();
    p.guu = sg_.guu();
    p.guv = sg_.guv();
    p.gvv = sg_.gvv();
    p.auu = sg_.auu();
    p.auv = sg_.auv();
    p.avv = sg_.avv();
    p.cosPer = sg_.cosPer();
    p.sinPer = sg_.sinPer();
    p.tanu = tanu_.data();
    p.tanv = tanv_.data();
    p.tanvPer = tanv_per_.data();
    p.bDotN = d_bdotn;
    p.wInt = w_int_.data();
    p.nZeta = sizes_.nZeta;
    p.nThetaEven = sizes_.nThetaEven;
    p.nZnT = sizes_.nZnT;
    p.nfp = sizes_.nfp;
    p.nvper = nvper_;
    p.axisymmetric = (sizes_.nZeta == 1);
    p.greenp = greenp_.data();
    p.gstore = gstore_.data();

    if (p.axisymmetric) {
        launchChecked(
            reinterpret_cast<const void*>(&regularizedGreenpAxisymKernel<T>),
            sizes_.nZnT * sizes_.nThetaEven, kBlockSize, p, stream_,
            "RegularizedIntegralsOperator::update: greenp (axisym)");
        launchChecked(
            reinterpret_cast<const void*>(&regularizedGstoreAxisymKernel<T>),
            sizes_.nThetaEven, kBlockSize, p, stream_,
            "RegularizedIntegralsOperator::update: gstore (axisym)");
    } else {
        launchChecked(
            reinterpret_cast<const void*>(&regularizedGreenpKernel<T>),
            sizes_.nZnT * sizes_.nThetaEven * sizes_.nZeta, kBlockSize, p,
            stream_, "RegularizedIntegralsOperator::update: greenp");
        launchChecked(
            reinterpret_cast<const void*>(&regularizedGstoreKernel<T>),
            sizes_.nThetaEven * sizes_.nZeta, kBlockSize, p, stream_,
            "RegularizedIntegralsOperator::update: gstore");
    }
}

}  // namespace vfield

#endif  // VFIELD_SRC_REGULARIZED_INTEGRALS_IMPL_CUH_
