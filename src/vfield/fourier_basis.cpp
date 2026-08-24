// fourier_basis.cpp — host-side Fourier basis tables (NESTOR layout).
//
// Direct port of vmecpp's FourierBasis::compute_fourier_basis and the
// cos/sin <-> cc/ss/sc/cs conversion functions (common/fourier_basis/
// fourier_basis.cc), restricted to the FourierBasisFastToroidalLayout.
// The accumulation order and normalization factors are kept verbatim: the
// golden tests compare against them.
#include "vfield/common/fourier_basis.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace vfield {

FourierBasis::FourierBasis(const Sizes& sizes) : sizes_(sizes) {
    mscale.resize(sizes_.mnyq2 + 1);
    nscale.resize(sizes_.nnyq2 + 1);

    cosmu.resize(sizes_.nThetaReduced * (sizes_.mnyq2 + 1));
    sinmu.resize(sizes_.nThetaReduced * (sizes_.mnyq2 + 1));
    cosmum.resize(sizes_.nThetaReduced * (sizes_.mnyq2 + 1));
    sinmum.resize(sizes_.nThetaReduced * (sizes_.mnyq2 + 1));
    cosmui.resize(sizes_.nThetaReduced * (sizes_.mnyq2 + 1));
    sinmui.resize(sizes_.nThetaReduced * (sizes_.mnyq2 + 1));

    cosnv.resize((sizes_.nnyq2 + 1) * sizes_.nZeta);
    sinnv.resize((sizes_.nnyq2 + 1) * sizes_.nZeta);
    cosnvn.resize((sizes_.nnyq2 + 1) * sizes_.nZeta);
    sinnvn.resize((sizes_.nnyq2 + 1) * sizes_.nZeta);

    compute_fourier_basis(sizes_.nfp);
}

void FourierBasis::compute_fourier_basis(int nfp) {
    static constexpr double TWO_PI = 2.0 * std::numbers::pi;

    // Fourier transforms are always computed in VMEC
    // over the reduced theta interval from [0, pi].
    // Thus, need a fixed normalization factor (cannot use dnorm3 or wInt in
    // Sizes) here.
    const double intNorm = 1.0 / (sizes_.nZeta * (sizes_.nThetaReduced - 1));

    // poloidal
    for (int m = 0; m < sizes_.mnyq2 + 1; ++m) {
        // DFTs for m>0 need 1/pi==2/(2pi) normalization factor
        // vs. 1/(2pi) for the cos(m=0)-mode.
        // --> introduce one sqrt(2) in fwd-DFT (geometry-into-realspace)
        //     and one sqrt(2) into inv-DFT (forces-into-Fourier) via mscale
        if (m == 0) {
            mscale[m] = 1.0;
        } else {
            mscale[m] = std::numbers::sqrt2;
        }
    }  // m

    for (int m = 0; m < sizes_.mnyq2 + 1; ++m) {
        for (int l = 0; l < sizes_.nThetaReduced; ++l) {
            // need to compute theta grid using _full_ number of theta points!
            const double theta = TWO_PI * l / sizes_.nThetaEven;
            const int idx_ml = poloidal_basis_index(m, l, sizes_.mnyq2 + 1);

            const double arg = m * theta;

            // poloidal Fourier basis
            cosmu[idx_ml] = std::cos(arg) * mscale[m];
            sinmu[idx_ml] = std::sin(arg) * mscale[m];

            // integration
            cosmui[idx_ml] = cosmu[idx_ml] * intNorm;
            sinmui[idx_ml] = sinmu[idx_ml] * intNorm;

            if (l == 0 || l == sizes_.nThetaReduced - 1) {
                cosmui[idx_ml] /= 2.0;
            }

            // poloidal derivatives
            cosmum[idx_ml] = m * cosmu[idx_ml];
            sinmum[idx_ml] = -m * sinmu[idx_ml];
        }  // l
    }  // m

    // toroidal
    for (int n = 0; n < sizes_.nnyq2 + 1; ++n) {
        // DFTs for m>0 need 1/pi==2/(2pi) normalization factor
        // vs. 1/(2pi) for the cos(m=0)-mode.
        // --> introduce one sqrt(2) in fwd-DFT (geometry-into-realspace)
        //     and one sqrt(2) into inv-DFT (forces-into-Fourier) via nscale
        if (n == 0) {
            nscale[n] = 1.0;
        } else {
            nscale[n] = std::numbers::sqrt2;
        }
    }  // n

    for (int k = 0; k < sizes_.nZeta; ++k) {
        const double zeta = TWO_PI * k / sizes_.nZeta;
        for (int n = 0; n < sizes_.nnyq2 + 1; ++n) {
            const int idx_kn = toroidal_basis_index(n, k, sizes_.nZeta);

            const double arg = n * zeta;

            // toroidal Fourier basis
            cosnv[idx_kn] = std::cos(arg) * nscale[n];
            sinnv[idx_kn] = std::sin(arg) * nscale[n];

            // toroidal derivatives
            cosnvn[idx_kn] = n * nfp * cosnv[idx_kn];
            sinnvn[idx_kn] = -n * nfp * sinnv[idx_kn];
        }  // n
    }  // k
}

// convert cos(xm[mn] theta - xn[mn] zeta) into 2D FC array form
int FourierBasis::cos_to_cc_ss(const std::span<const double> fc_cos,
                            std::span<double> fc_cc,
                            std::span<double> fc_ss,
                            int n_size,
                            int m_size) const {
    // m = 0: n =  0, 1, ..., ntor --> ntor + 1
    // m > 0: n = -ntor, ..., ntor --> (mpol - 1) * (2 * ntor + 1)
    int mnmax = (n_size + 1) + (m_size - 1) * (2 * n_size + 1);

    std::fill_n(fc_cc.data(), m_size * (n_size + 1), 0.0);
    std::fill_n(fc_ss.data(), m_size * (n_size + 1), 0.0);

    int mn = 0;

    int m = 0;
    for (int n = 0; n < n_size + 1; ++n) {
        int abs_n = abs(n);

        double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

        double normed_fc = basis_norm * fc_cos[mn];

        fc_cc[product_index(m, abs_n, m_size)] += normed_fc;
        // no contribution to fc_ss where (m == 0 || n == 0)

        mn++;
    }

    for (m = 1; m < m_size; ++m) {
        for (int n = -n_size; n < n_size + 1; ++n) {
            int abs_n = abs(n);
            int sgn_n = signum(n);

            double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

            double normed_fc = basis_norm * fc_cos[mn];

            fc_cc[product_index(m, abs_n, m_size)] += normed_fc;
            if (abs_n > 0) {
                fc_ss[product_index(m, abs_n, m_size)] += sgn_n * normed_fc;
            }

            mn++;
        }  // n
    }  // m

    if (mn != mnmax) {
        throw std::runtime_error(
            "counting error in cos_to_cc_ss: mn=" + std::to_string(mn) +
            " should be " + std::to_string(mnmax));
    }

    return mnmax;
}

int FourierBasis::sin_to_sc_cs(const std::span<const double> fc_sin,
                            std::span<double> fc_sc,
                            std::span<double> fc_cs,
                            int n_size,
                            int m_size) const {
    // m = 0: n =  0, 1, ..., ntor --> ntor + 1
    // m > 0: n = -ntor, ..., ntor --> (mpol - 1) * (2 * ntor + 1)
    int mnmax = (n_size + 1) + (m_size - 1) * (2 * n_size + 1);

    std::fill_n(fc_sc.data(), m_size * (n_size + 1), 0.0);
    std::fill_n(fc_cs.data(), m_size * (n_size + 1), 0.0);

    int mn = 1;

    int m = 0;
    for (int n = 1; n < n_size + 1; ++n) {
        int abs_n = abs(n);
        int sgn_n = signum(n);

        double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

        double normed_fc = basis_norm * fc_sin[mn];

        // no contribution to fc_sc where m == 0
        // check for n > 0 is redundant when starting loop at n=1
        fc_cs[product_index(m, abs_n, m_size)] = -sgn_n * normed_fc;

        mn++;
    }

    for (m = 1; m < m_size; ++m) {
        for (int n = -n_size; n < n_size + 1; ++n) {
            int abs_n = abs(n);
            int sgn_n = signum(n);

            double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

            double normed_fc = basis_norm * fc_sin[mn];

            fc_sc[product_index(m, abs_n, m_size)] += normed_fc;
            if (abs_n > 0) {
                fc_cs[product_index(m, abs_n, m_size)] += -sgn_n * normed_fc;
            }

            mn++;
        }  // n
    }  // m

    if (mn != mnmax) {
        throw std::runtime_error(
            "counting error in sin_to_sc_cs: mn=" + std::to_string(mn) +
            " should be " + std::to_string(mnmax));
    }

    return mnmax;
}

int FourierBasis::cc_ss_to_cos(const std::span<const double> fc_cc,
                            const std::span<const double> fc_ss,
                            std::span<double> fc_cos,
                            int n_size,
                            int m_size) const {
    // m = 0: n =  0, 1, ..., ntor --> ntor + 1
    // m > 0: n = -ntor, ..., ntor --> (mpol - 1) * (2 * ntor + 1)
    int mnmax = (n_size + 1) + (m_size - 1) * (2 * n_size + 1);

    std::fill_n(fc_cos.data(), mnmax, 0.0);

    int mn = 0;

    int m = 0;
    for (int n = 0; n < n_size + 1; ++n) {
        double basis_norm = 1.0 / (mscale[m] * nscale[n]);

        fc_cos[mn] = fc_cc[product_index(m, n, m_size)] / basis_norm;

        mn++;
    }  // n

    for (m = 1; m < m_size; ++m) {
        for (int n = -n_size; n < n_size + 1; ++n) {
            int abs_n = abs(n);
            int sgn_n = signum(n);

            double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

            if (abs_n == 0) {
                fc_cos[mn] = fc_cc[product_index(m, abs_n, m_size)] / basis_norm;
            } else {
                double raw_cc = fc_cc[product_index(m, abs_n, m_size)];
                double raw_ss = fc_ss[product_index(m, abs_n, m_size)];
                fc_cos[mn] = 0.5 * (raw_cc + sgn_n * raw_ss) / basis_norm;
            }

            mn++;
        }  // n
    }  // m

    if (mn != mnmax) {
        throw std::runtime_error(
            "counting error in cc_ss_to_cos: mn=" + std::to_string(mn) +
            " should be " + std::to_string(mnmax));
    }

    return mnmax;
}

int FourierBasis::sc_cs_to_sin(const std::span<const double> fc_sc,
                            const std::span<const double> fc_cs,
                            std::span<double> fc_sin,
                            int n_size,
                            int m_size) const {
    // m = 0: n =  0, 1, ..., ntor --> ntor + 1
    // m > 0: n = -ntor, ..., ntor --> (mpol - 1) * (2 * ntor + 1)
    int mnmax = (n_size + 1) + (m_size - 1) * (2 * n_size + 1);

    std::fill_n(fc_sin.data(), mnmax, 0.0);

    int mn = 1;

    int m = 0;
    for (int n = 1; n < n_size + 1; ++n) {
        double basis_norm = 1.0 / (mscale[m] * nscale[n]);

        fc_sin[mn] = -fc_cs[product_index(m, n, m_size)] / basis_norm;

        mn++;
    }  // n

    for (m = 1; m < m_size; ++m) {
        for (int n = -n_size; n < n_size + 1; ++n) {
            int abs_n = abs(n);
            int sgn_n = signum(n);

            double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

            if (abs_n == 0) {
                fc_sin[mn] = fc_sc[product_index(m, abs_n, m_size)] / basis_norm;
            } else {
                double raw_sc = fc_sc[product_index(m, abs_n, m_size)];
                double raw_cs = fc_cs[product_index(m, abs_n, m_size)];
                fc_sin[mn] = 0.5 * (raw_sc - sgn_n * raw_cs) / basis_norm;
            }

            mn++;
        }  // n
    }  // m

    if (mn != mnmax) {
        throw std::runtime_error(
            "counting error in sc_cs_to_sin: mn=" + std::to_string(mn) +
            " should be " + std::to_string(mnmax));
    }

    return mnmax;
}

}  // namespace vfield
