// fourier_basis.cpp — host-side Fourier basis tables (NESTOR layout).
//
// Direct port of vmecpp's FourierBasis::computeFourierBasis and the
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

    computeFourierBasis(sizes_.nfp);
}

void FourierBasis::computeFourierBasis(int nfp) {
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
            const int idx_ml = poloidalBasisIndex(m, l, sizes_.mnyq2 + 1);

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
            const int idx_kn = toroidalBasisIndex(n, k, sizes_.nZeta);

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
int FourierBasis::cosToCcSs(const std::span<const double> fcCos,
                            std::span<double> fcCC,
                            std::span<double> fcSS,
                            int nSize,
                            int mSize) const {
    // m = 0: n =  0, 1, ..., ntor --> ntor + 1
    // m > 0: n = -ntor, ..., ntor --> (mpol - 1) * (2 * ntor + 1)
    int mnmax = (nSize + 1) + (mSize - 1) * (2 * nSize + 1);

    std::fill_n(fcCC.data(), mSize * (nSize + 1), 0.0);
    std::fill_n(fcSS.data(), mSize * (nSize + 1), 0.0);

    int mn = 0;

    int m = 0;
    for (int n = 0; n < nSize + 1; ++n) {
        int abs_n = abs(n);

        double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

        double normedFC = basis_norm * fcCos[mn];

        fcCC[productIndex(m, abs_n, mSize)] += normedFC;
        // no contribution to fcSS where (m == 0 || n == 0)

        mn++;
    }

    for (m = 1; m < mSize; ++m) {
        for (int n = -nSize; n < nSize + 1; ++n) {
            int abs_n = abs(n);
            int sgn_n = signum(n);

            double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

            double normedFC = basis_norm * fcCos[mn];

            fcCC[productIndex(m, abs_n, mSize)] += normedFC;
            if (abs_n > 0) {
                fcSS[productIndex(m, abs_n, mSize)] += sgn_n * normedFC;
            }

            mn++;
        }  // n
    }  // m

    if (mn != mnmax) {
        throw std::runtime_error(
            "counting error in cosToCcSs: mn=" + std::to_string(mn) +
            " should be " + std::to_string(mnmax));
    }

    return mnmax;
}

int FourierBasis::sinToScCs(const std::span<const double> fcSin,
                            std::span<double> fcSC,
                            std::span<double> fcCS,
                            int nSize,
                            int mSize) const {
    // m = 0: n =  0, 1, ..., ntor --> ntor + 1
    // m > 0: n = -ntor, ..., ntor --> (mpol - 1) * (2 * ntor + 1)
    int mnmax = (nSize + 1) + (mSize - 1) * (2 * nSize + 1);

    std::fill_n(fcSC.data(), mSize * (nSize + 1), 0.0);
    std::fill_n(fcCS.data(), mSize * (nSize + 1), 0.0);

    int mn = 1;

    int m = 0;
    for (int n = 1; n < nSize + 1; ++n) {
        int abs_n = abs(n);
        int sgn_n = signum(n);

        double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

        double normedFC = basis_norm * fcSin[mn];

        // no contribution to fcSC where m == 0
        // check for n > 0 is redundant when starting loop at n=1
        fcCS[productIndex(m, abs_n, mSize)] = -sgn_n * normedFC;

        mn++;
    }

    for (m = 1; m < mSize; ++m) {
        for (int n = -nSize; n < nSize + 1; ++n) {
            int abs_n = abs(n);
            int sgn_n = signum(n);

            double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

            double normedFC = basis_norm * fcSin[mn];

            fcSC[productIndex(m, abs_n, mSize)] += normedFC;
            if (abs_n > 0) {
                fcCS[productIndex(m, abs_n, mSize)] += -sgn_n * normedFC;
            }

            mn++;
        }  // n
    }  // m

    if (mn != mnmax) {
        throw std::runtime_error(
            "counting error in sinToScCs: mn=" + std::to_string(mn) +
            " should be " + std::to_string(mnmax));
    }

    return mnmax;
}

int FourierBasis::ccSsToCos(const std::span<const double> fcCC,
                            const std::span<const double> fcSS,
                            std::span<double> fcCos,
                            int nSize,
                            int mSize) const {
    // m = 0: n =  0, 1, ..., ntor --> ntor + 1
    // m > 0: n = -ntor, ..., ntor --> (mpol - 1) * (2 * ntor + 1)
    int mnmax = (nSize + 1) + (mSize - 1) * (2 * nSize + 1);

    std::fill_n(fcCos.data(), mnmax, 0.0);

    int mn = 0;

    int m = 0;
    for (int n = 0; n < nSize + 1; ++n) {
        double basis_norm = 1.0 / (mscale[m] * nscale[n]);

        fcCos[mn] = fcCC[productIndex(m, n, mSize)] / basis_norm;

        mn++;
    }  // n

    for (m = 1; m < mSize; ++m) {
        for (int n = -nSize; n < nSize + 1; ++n) {
            int abs_n = abs(n);
            int sgn_n = signum(n);

            double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

            if (abs_n == 0) {
                fcCos[mn] = fcCC[productIndex(m, abs_n, mSize)] / basis_norm;
            } else {
                double raw_cc = fcCC[productIndex(m, abs_n, mSize)];
                double raw_ss = fcSS[productIndex(m, abs_n, mSize)];
                fcCos[mn] = 0.5 * (raw_cc + sgn_n * raw_ss) / basis_norm;
            }

            mn++;
        }  // n
    }  // m

    if (mn != mnmax) {
        throw std::runtime_error(
            "counting error in ccSsToCos: mn=" + std::to_string(mn) +
            " should be " + std::to_string(mnmax));
    }

    return mnmax;
}

int FourierBasis::scCsToSin(const std::span<const double> fcSC,
                            const std::span<const double> fcCS,
                            std::span<double> fcSin,
                            int nSize,
                            int mSize) const {
    // m = 0: n =  0, 1, ..., ntor --> ntor + 1
    // m > 0: n = -ntor, ..., ntor --> (mpol - 1) * (2 * ntor + 1)
    int mnmax = (nSize + 1) + (mSize - 1) * (2 * nSize + 1);

    std::fill_n(fcSin.data(), mnmax, 0.0);

    int mn = 1;

    int m = 0;
    for (int n = 1; n < nSize + 1; ++n) {
        double basis_norm = 1.0 / (mscale[m] * nscale[n]);

        fcSin[mn] = -fcCS[productIndex(m, n, mSize)] / basis_norm;

        mn++;
    }  // n

    for (m = 1; m < mSize; ++m) {
        for (int n = -nSize; n < nSize + 1; ++n) {
            int abs_n = abs(n);
            int sgn_n = signum(n);

            double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

            if (abs_n == 0) {
                fcSin[mn] = fcSC[productIndex(m, abs_n, mSize)] / basis_norm;
            } else {
                double raw_sc = fcSC[productIndex(m, abs_n, mSize)];
                double raw_cs = fcCS[productIndex(m, abs_n, mSize)];
                fcSin[mn] = 0.5 * (raw_sc - sgn_n * raw_cs) / basis_norm;
            }

            mn++;
        }  // n
    }  // m

    if (mn != mnmax) {
        throw std::runtime_error(
            "counting error in scCsToSin: mn=" + std::to_string(mn) +
            " should be " + std::to_string(mnmax));
    }

    return mnmax;
}

}  // namespace vfield
