// test_fourier_basis.cpp — FourierBasis scale/orthogonality/round-trip checks.
//
// Host-only. Verifies the mscale/nscale normalization split, the integration
// orthogonality of the basis tables, the derivative tables against direct
// differentiation, and the cos<->cc/ss, sin<->sc/cs round-trips.
#include "vfield/common/fourier_basis.hpp"
#include "vfield_test.h"

#include <cmath>
#include <numbers>
#include <vector>

using vfield::FourierBasis;
using vfield::Sizes;
using vfield::test::check;
using vfield::test::expect_near;
using vfield::test::max_diff;
using vfield::test::summary;

int main() {
    // cth-like resolution: nfp=5, mpol=5, ntor=4, ntheta=16, nzeta=36.
    Sizes sizes(false, 5, 5, 4, 16, 36);
    FourierBasis fb(sizes);

    // mscale/nscale: 1 for the zero mode, sqrt(2) otherwise.
    expect_near(fb.mscale[0], 1.0, 1e-15, "mscale[0] == 1");
    expect_near(fb.nscale[0], 1.0, 1e-15, "nscale[0] == 1");
    bool scales_ok = true;
    for (int m = 1; m <= sizes.mnyq2; ++m) {
        if (std::fabs(fb.mscale[m] - std::numbers::sqrt2) > 1e-15)
            scales_ok = false;
    }
    for (int n = 1; n <= sizes.nnyq2; ++n) {
        if (std::fabs(fb.nscale[n] - std::numbers::sqrt2) > 1e-15)
            scales_ok = false;
    }
    check(scales_ok, "mscale/nscale == sqrt(2) for nonzero modes");

    // Basis values against direct trigonometric evaluation.
    double max_err = 0.0;
    for (int m = 0; m <= sizes.mnyq2; ++m) {
        for (int l = 0; l < sizes.nThetaReduced; ++l) {
            const double theta = 2.0 * std::numbers::pi * l / sizes.nThetaEven;
            const int idx_ml =
                FourierBasis::poloidal_basis_index(m, l, sizes.mnyq2 + 1);
            max_err = std::max(max_err,
                               std::fabs(fb.cosmu[idx_ml] -
                                         std::cos(m * theta) * fb.mscale[m]));
            max_err = std::max(max_err,
                               std::fabs(fb.sinmu[idx_ml] -
                                         std::sin(m * theta) * fb.mscale[m]));
        }
    }
    for (int k = 0; k < sizes.nZeta; ++k) {
        const double zeta = 2.0 * std::numbers::pi * k / sizes.nZeta;
        for (int n = 0; n <= sizes.nnyq2; ++n) {
            const int idx_kn =
                FourierBasis::toroidal_basis_index(n, k, sizes.nZeta);
            max_err =
                std::max(max_err, std::fabs(fb.cosnv[idx_kn] -
                                            std::cos(n * zeta) * fb.nscale[n]));
            max_err =
                std::max(max_err, std::fabs(fb.sinnv[idx_kn] -
                                            std::sin(n * zeta) * fb.nscale[n]));
        }
    }
    check(max_err < 1e-14, "basis values match direct trig evaluation");

    // Derivative tables: cosmum = m*cosmu, sinmum = -m*sinmu;
    // cosnvn = n*nfp*cosnv, sinnvn = -n*nfp*sinnv (d/dphi).
    max_err = 0.0;
    for (int m = 0; m <= sizes.mnyq2; ++m) {
        for (int l = 0; l < sizes.nThetaReduced; ++l) {
            const int idx =
                FourierBasis::poloidal_basis_index(m, l, sizes.mnyq2 + 1);
            max_err = std::max(max_err,
                               std::fabs(fb.cosmum[idx] - m * fb.cosmu[idx]));
            max_err = std::max(max_err,
                               std::fabs(fb.sinmum[idx] + m * fb.sinmu[idx]));
        }
    }
    for (int n = 0; n <= sizes.nnyq2; ++n) {
        for (int k = 0; k < sizes.nZeta; ++k) {
            const int idx = FourierBasis::toroidal_basis_index(n, k, sizes.nZeta);
            max_err = std::max(
                max_err,
                std::fabs(fb.cosnvn[idx] - n * sizes.nfp * fb.cosnv[idx]));
            max_err = std::max(
                max_err,
                std::fabs(fb.sinnvn[idx] + n * sizes.nfp * fb.sinnv[idx]));
        }
    }
    check(max_err < 1e-13, "derivative tables match m*cosmu / -n*nfp*sinnv");

    // Integration orthogonality: sum_l cosmui[m,l]*cosmu[m',l] should be
    // 1/nZeta on the diagonal and ~0 off the diagonal (scale factors cancel
    // the 1/2 trapezoidal normalization).
    for (int m = 0; m < 6; ++m) {
        for (int m2 = 0; m2 < 6; ++m2) {
            double sum = 0.0;
            for (int l = 0; l < sizes.nThetaReduced; ++l) {
                const int i1 =
                    FourierBasis::poloidal_basis_index(m, l, sizes.mnyq2 + 1);
                const int i2 =
                    FourierBasis::poloidal_basis_index(m2, l, sizes.mnyq2 + 1);
                sum += fb.cosmui[i1] * fb.cosmu[i2];
            }
            const double expect = (m == m2) ? 1.0 / sizes.nZeta : 0.0;
            if (std::fabs(sum - expect) > 1e-12) {
                std::cout << "  orthogonality (m=" << m << ", m2=" << m2
                          << "): " << sum << " vs " << expect << '\n';
                vfield::test::failures()++;
            }
        }
    }
    check(vfield::test::failures() == 0, "poloidal integration orthogonality");

    // cos <-> (cc, ss) round trip.
    const int n_size = sizes.ntor;
    const int m_size = sizes.mpol;
    const int mnmax = (n_size + 1) + (m_size - 1) * (2 * n_size + 1);
    std::vector<double> fc_cos(mnmax);
    for (int i = 0; i < mnmax; ++i) fc_cos[i] = 0.1 * (i + 1) + 0.001 * i * i;
    std::vector<double> fc_cc(m_size * (n_size + 1));
    std::vector<double> fc_ss(m_size * (n_size + 1));
    fb.cos_to_cc_ss(fc_cos, fc_cc, fc_ss, n_size, m_size);
    std::vector<double> back(mnmax);
    fb.cc_ss_to_cos(fc_cc, fc_ss, back, n_size, m_size);
    check(max_diff(fc_cos, back) < 1e-13, "cos <-> (cc, ss) round trip");

    // sin <-> (sc, cs) round trip (skip the mn=0 slot, which is unused).
    std::vector<double> fc_sin(mnmax);
    for (int i = 1; i < mnmax; ++i) fc_sin[i] = 0.05 * i + 0.002 * i * i;
    std::vector<double> fc_sc(m_size * (n_size + 1));
    std::vector<double> fc_cs(m_size * (n_size + 1));
    fb.sin_to_sc_cs(fc_sin, fc_sc, fc_cs, n_size, m_size);
    std::vector<double> back2(mnmax);
    fb.sc_cs_to_sin(fc_sc, fc_cs, back2, n_size, m_size);
    check(max_diff(fc_sin, back2) < 1e-13, "sin <-> (sc, cs) round trip");

    return summary();
}
