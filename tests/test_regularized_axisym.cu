// test_regularized_axisym.cu — axisymmetric (nZeta == 1) path vs CPU mirror.
//
// No Fortran golden exists for the nvper = 64 toroidal-image path, so the
// operator's axisymmetric kernels are compared against a direct CPU mirror
// of vmecpp's update_axisymmetric on a synthetic axisymmetric boundary
// (double math on the T inputs). Both scalar types are instantiated; the
// float leg compares at a relaxed tolerance.
#include "vfield/common/fourier_basis.hpp"
#include "vfield/common/fourier_basis_device.cuh"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/regularized_integrals_operator.hpp"
#include "vfield/free_boundary/surface_geometry_operator.hpp"
#include "vfield_test_cuda_helper.cuh"

#include <cmath>
#include <numbers>
#include <vector>

using vfield::FourierBasis;
using vfield::FourierBasisDevice;
using vfield::RegularizedIntegralsOperator;
using vfield::Sizes;
using vfield::SurfaceGeometryOperator;
using vfield::test::check;
using vfield::test::max_rel_diff;
using vfield::test::summary;
using vfield::test::to_device;
using vfield::test::to_host;

namespace {

constexpr int NFP = 1;
constexpr int MPOL = 4;
constexpr int NTOR = 0;  // axisymmetric
constexpr int NTHETA = 24;
constexpr int NZETA = 1;  // single toroidal plane -> nvper = 64 path
constexpr int SIGN_J = -1;

void make_coefficients(std::vector<double>* rcc, std::vector<double>* zsc) {
    const int mnsize = MPOL * (NTOR + 1);
    rcc->assign(mnsize, 0.0);
    zsc->assign(mnsize, 0.0);
    double phase = 0.0;
    for (int m = 0; m < MPOL; ++m) {
        phase += 0.63;
        (*rcc)[m] = 0.8 + 0.06 * (m + 1) * std::cos(phase);
        (*zsc)[m] = 0.5 * std::sin(phase) + 0.04 * m;
    }
}

// CPU mirror of vmecpp's update_axisymmetric (nZeta == 1): sums over nvper
// toroidal images with the analytic approximation subtracted at every image,
// skipping the exact singularity. Computed in double from the T inputs.
void cpu_axisym_reference(const Sizes& s,
                        const std::vector<double>& r1b,
                        const std::vector<double>& z1b,
                        const std::vector<double>& rcosuv,
                        const std::vector<double>& rsinuv,
                        const std::vector<double>& rzb2,
                        const std::vector<double>& drv,
                        const std::vector<double>& snr,
                        const std::vector<double>& snv,
                        const std::vector<double>& snz,
                        const std::vector<double>& guu,
                        const std::vector<double>& guv,
                        const std::vector<double>& gvv,
                        const std::vector<double>& auu,
                        const std::vector<double>& auv,
                        const std::vector<double>& avv,
                        const std::vector<double>& bdotn,
                        std::vector<double>* greenp,
                        std::vector<double>* gstore) {
    const int nvper = 64;
    const double measure = 2.0 * std::numbers::pi / nvper;

    std::vector<double> tanu(s.nThetaEven);
    for (int l = 0; l < s.nThetaEven; ++l) {
        const double argu = std::numbers::pi / s.nThetaEven * l;
        if (std::abs(argu - 0.5 * std::numbers::pi) < 1e-15) {
            tanu[l] = 1e50;
        } else {
            tanu[l] = 2.0 * std::tan(argu);
        }
    }
    std::vector<double> tanv_per(nvper);
    for (int p = 0; p < nvper; ++p) {
        const double argv = std::numbers::pi * p / nvper;
        if (std::abs(argv - 0.5 * std::numbers::pi) < 1e-15) {
            tanv_per[p] = 1e50;
        } else {
            tanv_per[p] = 2.0 * std::tan(argv);
        }
    }

    greenp->assign(s.nZnT * s.nThetaEven, 0.0);
    gstore->assign(s.nThetaEven, 0.0);

    for (int klp = 0; klp < s.nZnT; ++klp) {
        const int ip_base = klp * s.nThetaEven;
        const double bexni = bdotn[klp] * s.wInt[klp];
        const double xp = rcosuv[klp];
        const double yp = rsinuv[klp];

        std::vector<double> gsave(s.nThetaEven);
        std::vector<double> dsave(s.nThetaEven);
        for (int kl = 0; kl < s.nThetaEven; ++kl) {
            gsave[kl] = rzb2[klp] + rzb2[kl] - 2 * z1b[kl] * z1b[klp];
            dsave[kl] = drv[klp] + z1b[kl] * snz[klp];
        }

        for (int p = 0; p < nvper; ++p) {
            const double cosper = std::cos(measure * p);
            const double sinper = std::sin(measure * p);
            const double xper = xp * cosper - yp * sinper;
            const double yper = xp * sinper + yp * cosper;
            const double sxsave =
                (snr[klp] * xper - snv[klp] * yper) / r1b[klp];
            const double sysave =
                (snr[klp] * yper + snv[klp] * xper) / r1b[klp];
            const double tanv_p = tanv_per[p];

            for (int kl = 0; kl < s.nThetaEven; ++kl) {
                if (p == 0 && kl == klp) continue;

                const int delta_l = (kl - klp + s.nThetaEven) % s.nThetaEven;

                double ga1 = guu[klp] * tanu[delta_l] * tanu[delta_l] +
                             guv[klp] * tanu[delta_l] * tanv_p +
                             gvv[klp] * tanv_p * tanv_p;
                double ga2 = auu[klp] * tanu[delta_l] * tanu[delta_l] +
                             auv[klp] * tanu[delta_l] * tanv_p +
                             avv[klp] * tanv_p * tanv_p;
                ga2 /= ga1;
                ga1 = 1.0 / std::sqrt(ga1);

                const double ftemp =
                    1.0 /
                    (gsave[kl] - 2 * (xper * rcosuv[kl] + yper * rsinuv[kl]));
                const double htemp = std::sqrt(ftemp);

                (*greenp)[ip_base + kl] +=
                    measure * (htemp * ftemp *
                                   (rcosuv[kl] * sxsave + rsinuv[kl] * sysave +
                                    dsave[kl]) -
                               ga1 * ga2);
                (*gstore)[kl] += bexni * measure * (htemp - ga1);
            }
        }
    }
}

// Convert the (possibly float) device-downloaded arrays to double for the
// CPU mirror (which computes in double from the T inputs, the cuMES CPU-mirror
// pattern).
template <class T>
std::vector<double> as_double(const std::vector<T>& v) {
    return std::vector<double>(v.begin(), v.end());
}

template <class T>
void run_precision(double tol) {
    Sizes sizes(false, NFP, MPOL, NTOR, NTHETA, NZETA);
    FourierBasis fb(sizes);
    FourierBasisDevice<T> fbd(fb, sizes.lasym, sizes.nThetaEven);
    SurfaceGeometryOperator<T> sg(sizes, fbd);
    RegularizedIntegralsOperator<T> ri(sizes, sg);

    std::vector<double> rcc, zsc;
    make_coefficients(&rcc, &zsc);
    std::vector<double> rss(sizes.mnsize, 0.0);
    std::vector<double> zcs(sizes.mnsize, 0.0);

    std::vector<T> rcc_t(rcc.begin(), rcc.end());
    std::vector<T> zsc_t(zsc.begin(), zsc.end());
    std::vector<T> zero_t(sizes.mnsize, T(0));
    auto d_rcc = to_device(rcc_t);
    auto d_zsc = to_device(zsc_t);
    auto d_zero = to_device(zero_t);

    sg.update(d_rcc.data(), d_zero.data(), nullptr, nullptr, d_zsc.data(),
              d_zero.data(), nullptr, nullptr, SIGN_J, true);

    // Fabricate a smooth b_dot_n (any input works: the operator is linear in
    // it).
    std::vector<T> bdotn(sizes.nZnT);
    for (int kl = 0; kl < sizes.nZnT; ++kl) {
        bdotn[kl] = static_cast<T>(0.5 + 0.1 * std::sin(kl * 0.7));
    }
    auto d_bdotn = to_device(bdotn);
    ri.update(d_bdotn.data());

    // Download the geometry for the CPU mirror (double math).
    const auto r1b = to_host(sg.r1b(), sizes.nThetaEven * sizes.nZeta);
    const auto z1b = to_host(sg.z1b(), sizes.nThetaEven * sizes.nZeta);
    const auto rcosuv = to_host(sg.rcosuv(), sizes.nThetaEven * sizes.nZeta);
    const auto rsinuv = to_host(sg.rsinuv(), sizes.nThetaEven * sizes.nZeta);
    const auto rzb2 = to_host(sg.rzb2(), sizes.nThetaEven * sizes.nZeta);
    const auto drv = to_host(sg.drv(), sizes.nZnT);
    const auto snr = to_host(sg.snr(), sizes.nZnT);
    const auto snv = to_host(sg.snv(), sizes.nZnT);
    const auto snz = to_host(sg.snz(), sizes.nZnT);
    const auto guu = to_host(sg.guu(), sizes.nZnT);
    const auto guv = to_host(sg.guv(), sizes.nZnT);
    const auto gvv = to_host(sg.gvv(), sizes.nZnT);
    const auto auu = to_host(sg.auu(), sizes.nZnT);
    const auto auv = to_host(sg.auv(), sizes.nZnT);
    const auto avv = to_host(sg.avv(), sizes.nZnT);
    std::vector<double> bdotn_d(bdotn.begin(), bdotn.end());

    std::vector<double> greenp_ref, gstore_ref;
    cpu_axisym_reference(sizes, as_double(r1b), as_double(z1b), as_double(rcosuv),
                       as_double(rsinuv), as_double(rzb2), as_double(drv),
                       as_double(snr), as_double(snv), as_double(snz),
                       as_double(guu), as_double(guv), as_double(gvv),
                       as_double(auu), as_double(auv), as_double(avv), bdotn_d,
                       &greenp_ref, &gstore_ref);

    check(max_rel_diff(to_host(ri.greenp(), sizes.nZnT * sizes.nThetaEven),
                       greenp_ref) < tol,
          "greenp (axisym)");
    check(max_rel_diff(to_host(ri.gstore(), sizes.nThetaEven), gstore_ref) < tol,
          "gstore (axisym)");
}

}  // namespace

int main() {
    // The double tolerance is 1e-11: the GPU kernels contract multiply-adds
    // (FMA) while the CPU mirror does not, and greenp's
    // (htemp*ftemp*(...) - ga1*ga2) terms cancel, so the observed kernel-vs-
    // mirror gap is ~1e-12 — FMA noise, not a defect (measured 1.35e-12 at
    // tol 1e-12).
    run_precision<double>(1e-11);
    run_precision<float>(1e-4);
    return summary();
}
