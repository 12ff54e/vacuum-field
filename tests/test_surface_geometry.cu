// test_surface_geometry.cu — surface geometry vs direct trig reference.
//
// Builds a synthetic (m,n)-rich boundary, evaluates SurfaceGeometryOperator
// on the GPU and a direct trigonometric CPU reference (double math on the T
// inputs — the cuMES CPU-mirror pattern), and compares every output array
// over the full surface / reduced poloidal range. Both scalar types are
// instantiated; the float leg compares at a relaxed tolerance.
#include "vfield/common/fourier_basis.hpp"
#include "vfield/common/fourier_basis_device.cuh"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/surface_geometry_operator.hpp"
#include "vfield_test_cuda_helper.cuh"

#include <cmath>
#include <numbers>
#include <vector>

using vfield::FourierBasis;
using vfield::FourierBasisDevice;
using vfield::Sizes;
using vfield::SurfaceGeometryOperator;
using vfield::test::check;
using vfield::test::max_rel_diff;
using vfield::test::summary;
using vfield::test::toDevice;
using vfield::test::toHost;

namespace {

constexpr int NFP = 5;
constexpr int MPOL = 5;
constexpr int NTOR = 4;
constexpr int NTHETA = 16;
constexpr int NZETA = 36;
constexpr int SIGN_J = -1;

// Synthetic boundary coefficients with every (m,n) slot occupied.
void makeCoefficients(std::vector<double>* rcc,
                      std::vector<double>* rss,
                      std::vector<double>* zsc,
                      std::vector<double>* zcs) {
    const int mnsize = MPOL * (NTOR + 1);
    rcc->assign(mnsize, 0.0);
    rss->assign(mnsize, 0.0);
    zsc->assign(mnsize, 0.0);
    zcs->assign(mnsize, 0.0);
    double phase = 0.0;
    for (int n = 0; n <= NTOR; ++n) {
        for (int m = 0; m < MPOL; ++m) {
            const int idx = n * MPOL + m;
            phase += 0.37;
            (*rcc)[idx] = 0.9 + 0.03 * (m + 1) * std::cos(phase) + 0.01 * n;
            (*rss)[idx] = 0.05 * std::sin(phase * 2.0) + 0.02 * m;
            (*zsc)[idx] = 0.7 * std::sin(phase) + 0.02 * n * m;
            (*zcs)[idx] = 0.04 * std::cos(phase * 1.7);
        }
    }
}

// Direct trigonometric evaluation of R, Z and their first/second derivatives
// over the FULL poloidal range, in double, plus the derived quantities in
// vmecpp's conventions (guv = 2*g_theta_phi/nfp, gvv = g_phi_phi/nfp^2,
// sn = signJ * (-r*N)).
struct CpuSurface {
    std::vector<double> r1b, z1b, rcosuv, rsinuv, rzb2;
    std::vector<double> rub, rvb, zub, zvb;
    std::vector<double> ruu, ruv, rvv, zuu, zuv, zvv;
    std::vector<double> snr, snv, snz, guu, guv, gvv, auu, auv, avv, drv;
};

CpuSurface cpuReference(const std::vector<double>& rcc,
                        const std::vector<double>& rss,
                        const std::vector<double>& zsc,
                        const std::vector<double>& zcs,
                        const Sizes& s) {
    CpuSurface out;
    const int n_full = s.nThetaEven * s.nZeta;
    out.r1b.assign(n_full, 0.0);
    out.z1b.assign(n_full, 0.0);
    out.rub.assign(n_full, 0.0);
    out.rvb.assign(n_full, 0.0);
    out.zub.assign(n_full, 0.0);
    out.zvb.assign(n_full, 0.0);
    out.ruu.assign(n_full, 0.0);
    out.ruv.assign(n_full, 0.0);
    out.rvv.assign(n_full, 0.0);
    out.zuu.assign(n_full, 0.0);
    out.zuv.assign(n_full, 0.0);
    out.zvv.assign(n_full, 0.0);

    for (int l = 0; l < s.nThetaEven; ++l) {
        const double theta = 2.0 * std::numbers::pi * l / s.nThetaEven;
        for (int k = 0; k < s.nZeta; ++k) {
            const double zeta = 2.0 * std::numbers::pi * k / s.nZeta;
            const int kl = l * s.nZeta + k;

            double r = 0, z = 0, rt = 0, zt = 0, rp = 0, zp = 0;
            double rtt = 0, ztt = 0, rtp = 0, ztp = 0, rpp = 0, zpp = 0;
            for (int n = 0; n <= s.ntor; ++n) {
                for (int m = 0; m < s.mpol; ++m) {
                    const int idx = n * s.mpol + m;
                    // The basis carries the DFT normalization split
                    // (mscale/nscale), so the real-space series uses the
                    // scaled trig values.
                    const double scale = (m == 0 ? 1.0 : std::numbers::sqrt2) *
                                         (n == 0 ? 1.0 : std::numbers::sqrt2);
                    const double cm = std::cos(m * theta) * scale;
                    const double sm = std::sin(m * theta) * scale;
                    const double cn = std::cos(n * zeta);
                    const double sn = std::sin(n * zeta);
                    const double nnp = n * s.nfp;

                    r += rcc[idx] * cm * cn + rss[idx] * sm * sn;
                    z += zsc[idx] * sm * cn + zcs[idx] * cm * sn;
                    rt += rcc[idx] * (-m * sm) * cn + rss[idx] * (m * cm) * sn;
                    zt += zsc[idx] * (m * cm) * cn + zcs[idx] * (-m * sm) * sn;
                    rp += rcc[idx] * cm * (-nnp * sn) +
                          rss[idx] * sm * (nnp * cn);
                    zp += zsc[idx] * sm * (-nnp * sn) +
                          zcs[idx] * cm * (nnp * cn);
                    rtt += rcc[idx] * (-m * m * cm) * cn +
                           rss[idx] * (-m * m * sm) * sn;
                    ztt += zsc[idx] * (-m * m * sm) * cn +
                           zcs[idx] * (-m * m * cm) * sn;
                    rtp += rcc[idx] * (-m * sm) * (-nnp * sn) +
                           rss[idx] * (m * cm) * (nnp * cn);
                    ztp += zsc[idx] * (m * cm) * (-nnp * sn) +
                           zcs[idx] * (-m * sm) * (nnp * cn);
                    rpp += rcc[idx] * cm * (-nnp * nnp * cn) +
                           rss[idx] * sm * (-nnp * nnp * sn);
                    zpp += zsc[idx] * sm * (-nnp * nnp * cn) +
                           zcs[idx] * cm * (-nnp * nnp * sn);
                }
            }

            out.r1b[kl] = r;
            out.z1b[kl] = z;
            out.rub[kl] = rt;
            out.zub[kl] = zt;
            out.rvb[kl] = rp;
            out.zvb[kl] = zp;
            out.ruu[kl] = rtt;
            out.zuu[kl] = ztt;
            out.ruv[kl] = rtp;
            out.zuv[kl] = ztp;
            out.rvv[kl] = rpp;
            out.zvv[kl] = zpp;
        }
    }

    out.snr.assign(s.nZnT, 0.0);
    out.snv.assign(s.nZnT, 0.0);
    out.snz.assign(s.nZnT, 0.0);
    out.guu.assign(s.nZnT, 0.0);
    out.guv.assign(s.nZnT, 0.0);
    out.gvv.assign(s.nZnT, 0.0);
    out.auu.assign(s.nZnT, 0.0);
    out.auv.assign(s.nZnT, 0.0);
    out.avv.assign(s.nZnT, 0.0);
    out.drv.assign(s.nZnT, 0.0);
    out.rcosuv.assign(n_full, 0.0);
    out.rsinuv.assign(n_full, 0.0);
    out.rzb2.assign(n_full, 0.0);

    for (int l = 0; l < s.nThetaEff; ++l) {
        for (int k = 0; k < s.nZeta; ++k) {
            const int kl = l * s.nZeta + k;
            const double r = out.r1b[kl];
            const double z = out.z1b[kl];
            const double rt = out.rub[kl];
            const double rp = out.rvb[kl];
            const double zt = out.zub[kl];
            const double zp = out.zvb[kl];
            const double sign_j = SIGN_J;

            const double snr = sign_j * r * zt;
            const double snv = sign_j * (rt * zp - zt * rp);
            const double snz = -sign_j * r * rt;

            out.snr[kl] = snr;
            out.snv[kl] = snv;
            out.snz[kl] = snz;
            out.guu[kl] = rt * rt + zt * zt;
            out.guv[kl] = 2.0 * (rt * rp + zt * zp) / s.nfp;
            out.gvv[kl] = (rp * rp + r * r + zp * zp) / (s.nfp * s.nfp);
            out.auu[kl] = (out.ruu[kl] * snr + out.zuu[kl] * snz) / 2;
            out.auv[kl] =
                (out.ruv[kl] * snr + rt * snv + out.zuv[kl] * snz) / s.nfp;
            out.avv[kl] =
                (rp * snv + ((out.rvv[kl] - r) * snr + out.zvv[kl] * snz) / 2) /
                (s.nfp * s.nfp);
            out.drv[kl] = -(r * snr + z * snz);
        }
    }

    for (int kl = 0; kl < n_full; ++kl) {
        const int k = kl % s.nZeta;
        const double phi = 2.0 * std::numbers::pi * k / (s.nfp * s.nZeta);
        out.rzb2[kl] = out.r1b[kl] * out.r1b[kl] + out.z1b[kl] * out.z1b[kl];
        out.rcosuv[kl] = out.r1b[kl] * std::cos(phi);
        out.rsinuv[kl] = out.r1b[kl] * std::sin(phi);
    }

    return out;
}

template <class T>
void runPrecision(double tol) {
    Sizes sizes(false, NFP, MPOL, NTOR, NTHETA, NZETA);
    FourierBasis fb(sizes);
    FourierBasisDevice<T> fbd(fb, sizes.lasym, sizes.nThetaEven);
    SurfaceGeometryOperator<T> sg(sizes, fbd);

    std::vector<double> rcc, rss, zsc, zcs;
    makeCoefficients(&rcc, &rss, &zsc, &zcs);
    std::vector<T> rcc_t(rcc.begin(), rcc.end());
    std::vector<T> rss_t(rss.begin(), rss.end());
    std::vector<T> zsc_t(zsc.begin(), zsc.end());
    std::vector<T> zcs_t(zcs.begin(), zcs.end());

    auto d_rcc = toDevice(rcc_t);
    auto d_rss = toDevice(rss_t);
    auto d_zsc = toDevice(zsc_t);
    auto d_zcs = toDevice(zcs_t);

    sg.update(d_rcc.data(), d_rss.data(), nullptr, nullptr, d_zsc.data(),
              d_zcs.data(), nullptr, nullptr, SIGN_J, true);

    const CpuSurface ref = cpuReference(rcc, rss, zsc, zcs, sizes);
    const int n_full = sizes.nThetaEven * sizes.nZeta;

    check(max_rel_diff(toHost(sg.r1b(), n_full), ref.r1b) < tol,
          "r1b full surface");
    check(max_rel_diff(toHost(sg.z1b(), n_full), ref.z1b) < tol,
          "z1b full surface (mirror)");
    check(max_rel_diff(toHost(sg.rcosuv(), n_full), ref.rcosuv) < tol,
          "rcosuv");
    check(max_rel_diff(toHost(sg.rsinuv(), n_full), ref.rsinuv) < tol,
          "rsinuv");
    check(max_rel_diff(toHost(sg.rzb2(), n_full), ref.rzb2) < tol, "rzb2");
    check(max_rel_diff(toHost(sg.rub(), sizes.nZnT), ref.rub) < tol, "rub");
    check(max_rel_diff(toHost(sg.rvb(), sizes.nZnT), ref.rvb) < tol, "rvb");
    check(max_rel_diff(toHost(sg.zub(), sizes.nZnT), ref.zub) < tol, "zub");
    check(max_rel_diff(toHost(sg.zvb(), sizes.nZnT), ref.zvb) < tol, "zvb");
    check(max_rel_diff(toHost(sg.ruu(), sizes.nZnT), ref.ruu) < tol, "ruu");
    check(max_rel_diff(toHost(sg.ruv(), sizes.nZnT), ref.ruv) < tol, "ruv");
    check(max_rel_diff(toHost(sg.rvv(), sizes.nZnT), ref.rvv) < tol, "rvv");
    check(max_rel_diff(toHost(sg.zuu(), sizes.nZnT), ref.zuu) < tol, "zuu");
    check(max_rel_diff(toHost(sg.zuv(), sizes.nZnT), ref.zuv) < tol, "zuv");
    check(max_rel_diff(toHost(sg.zvv(), sizes.nZnT), ref.zvv) < tol, "zvv");
    check(max_rel_diff(toHost(sg.snr(), sizes.nZnT), ref.snr) < tol, "snr");
    check(max_rel_diff(toHost(sg.snv(), sizes.nZnT), ref.snv) < tol, "snv");
    check(max_rel_diff(toHost(sg.snz(), sizes.nZnT), ref.snz) < tol, "snz");
    check(max_rel_diff(toHost(sg.guu(), sizes.nZnT), ref.guu) < tol, "guu");
    check(max_rel_diff(toHost(sg.guv(), sizes.nZnT), ref.guv) < tol, "guv");
    check(max_rel_diff(toHost(sg.gvv(), sizes.nZnT), ref.gvv) < tol, "gvv");
    check(max_rel_diff(toHost(sg.auu(), sizes.nZnT), ref.auu) < tol, "auu");
    check(max_rel_diff(toHost(sg.auv(), sizes.nZnT), ref.auv) < tol, "auv");
    check(max_rel_diff(toHost(sg.avv(), sizes.nZnT), ref.avv) < tol, "avv");
    check(max_rel_diff(toHost(sg.drv(), sizes.nZnT), ref.drv) < tol, "drv");

    // Also exercise the no-full-update path (first derivatives only).
    sg.update(d_rcc.data(), d_rss.data(), nullptr, nullptr, d_zsc.data(),
              d_zcs.data(), nullptr, nullptr, SIGN_J, false);
    check(max_rel_diff(toHost(sg.rub(), sizes.nZnT), ref.rub) < tol,
          "rub (no full update)");
    check(max_rel_diff(toHost(sg.guv(), sizes.nZnT), ref.guv) < tol,
          "guv (no full update)");
}

}  // namespace

int main() {
    runPrecision<double>(1e-12);
    runPrecision<float>(1e-4);
    return summary();
}
