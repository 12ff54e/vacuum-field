// sizes.cpp — derived grid/mode counts and integration weights.
//
// Direct port of vmecpp's Sizes::compute_derived_sizes (common/sizes/sizes.cc)
// with the absl CHECKs replaced by std::invalid_argument. The count
// adjustments must stay verbatim: every golden comparison depends on them.
#include "vfield/common/sizes.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace vfield {

Sizes::Sizes(bool lasym, int nfp, int mpol, int ntor, int ntheta, int nzeta)
    : lasym(lasym),
      nfp(nfp),
      mpol(mpol),
      ntor(ntor),
      ntheta(ntheta),
      nZeta(nzeta) {
    compute_derived_sizes();
}

// Assuming that the key parameters defining the array sizes etc. have been
// set, compute the derived sizes like actual array sizes etc.
void Sizes::compute_derived_sizes() {
    if (nfp < 1) {
        throw std::invalid_argument(
            "input variable 'nfp' needs to be >= 1, but is " +
            std::to_string(nfp));
    }

    if (mpol < 1) {
        throw std::invalid_argument(
            "input variable 'mpol' needs to be >= 1, but is " +
            std::to_string(mpol));
    }

    if (ntor < 0) {
        throw std::invalid_argument(
            "input variable 'ntor' needs to be >= 0, but is " +
            std::to_string(ntor));
    }

    // ntheta
    if (ntheta < 2 * mpol + 6) { ntheta = 2 * mpol + 6; }

    // nzeta
    if (ntor == 0 && nZeta < 1) {
        // Tokamak (ntor=0) needs (at least) nzeta=1
        // I think this implies that (in principle, not reasonable) one could
        // do an axisymmetric run with nzeta > 1 ...
        nZeta = 1;
    }

    if (ntor > 0) {
        // 3D/Stellarator case needs Nyquist criterion fulfilled for nzeta
        // wrt. ntor
        if (nZeta < 2 * ntor + 4) { nZeta = 2 * ntor + 4; }
    }

    // derived

    // flag to indicate a three-dimensional case (== has toroidal variation)
    lthreed = (ntor > 0);

    // real-space array sizes

    // [0, 2pi[ --> EXCLUDING endpoint!
    nThetaEven = 2 * (ntheta / 2);

    // [0, pi] --> INCLUDING endpoint!
    nThetaReduced = nThetaEven / 2 + 1;

    if (lasym) {
        nThetaEff = nThetaEven;
    } else {
        // use stellarator- or up/down-symmetry
        // --> only eval on reduced [0, pi] poloidal interval
        nThetaEff = nThetaReduced;
    }

    // surface is always full in toroidal direction
    // but can be reduced in poloidal direction --> nTheta_Eff_
    nZnT = nZeta * nThetaEff;

    // normalization factor for poloidal integrals
    // default case: use stellarator symmetry
    // --> # of gaps between grid points is one less than number of grid
    // points (which INCLUDE endpoint in symmetric case)
    double dnorm3 = 1.0 / (nZeta * (nThetaReduced - 1));
    if (lasym) { dnorm3 = 1.0 / (nZeta * nThetaEven); }

    wInt.resize(nThetaEff);
    for (int l = 0; l < nThetaEff; ++l) {
        wInt[l] = dnorm3;
        if (!lasym && (l == 0 || l == nThetaReduced - 1)) {
            // weight back to 1 at the endpoints
            wInt[l] /= 2.0;
        }
    }

    mnsize = mpol * (ntor + 1);

    // m = 0: n =  0, 1, ..., ntor --> ntor + 1
    // m > 0: n = -ntor, ..., ntor --> (mpol - 1) * (2 * ntor + 1)
    mnmax = (ntor + 1) + (mpol - 1) * (2 * ntor + 1);

    // --------- Nyquist sizes

    // NEED 2 X NYQUIST FOR FAST HESSIAN CALCULATIONS
    // maximum mode numbers supported by grid
    int mnyq0 = nThetaEven / 2;
    int nnyq0 = nZeta / 2;

    // make sure that mnyq, nnyq are at least twice mpol-1, ntor
    // or large enough to fully represent the information held in realspace
    // (mnyq0, nnyq0)
    mnyq2 = std::max(0, std::max(2 * mnyq0, 2 * (mpol - 1)));
    nnyq2 = std::max(0, std::max(2 * nnyq0, 2 * ntor));
}

}  // namespace vfield
