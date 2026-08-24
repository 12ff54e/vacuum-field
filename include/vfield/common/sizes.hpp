// sizes.hpp — grid/mode counts and integration weights for the vacuum solver.
//
// A port of vmecpp's Sizes restricted to the fields the free-boundary modules
// consume. `compute_derived_sizes` reproduces vmecpp's count adjustments
// verbatim: ntheta is raised to at least 2*mpol+6 (Nyquist), nzeta to at
// least 2*ntor+4 in 3D (and forced to 1 for ntor == 0), the surface grid is
// [0, 2pi[ in poloidal angle on nThetaEven points with the symmetric case
// evaluated only on the reduced half [0, pi] (nThetaReduced points, endpoint
// included), and wInt carries the trapezoidal poloidal weights including the
// 1/nZeta normalization (endpoints halved in the symmetric case).
#ifndef VFIELD_COMMON_SIZES_HPP_
#define VFIELD_COMMON_SIZES_HPP_

#include <cstddef>
#include <vector>

namespace vfield {

struct Sizes {
    Sizes(bool lasym, int nfp, int mpol, int ntor, int ntheta, int nzeta);

    bool lasym;
    int nfp;
    int mpol;
    int ntor;
    int ntheta;  // possibly raised to >= 2*mpol+6
    int nZeta;   // possibly raised to >= 2*ntor+4 (or forced to 1)
    bool lthreed;

    // Surface grid: [0, 2pi[ excluding the endpoint on nThetaEven points;
    // [0, pi] including the endpoint on nThetaReduced points. The symmetric
    // case is evaluated only on the reduced half (nThetaEff = nThetaReduced).
    int nThetaEven;
    int nThetaReduced;
    int nThetaEff;
    int nZnT;  // nZeta * nThetaEff — surface grid points used by the solver

    // Trapezoidal poloidal integration weights (incl. 1/nZeta), [nThetaEff].
    std::vector<double> wInt;

    int mnsize;  // mpol * (ntor + 1) — NESTOR coefficient array size
    int mnmax;   // unique (m >= 0, |n| <= ntor, n >= 0 for m == 0) modes

    // Nyquist-extended mode ranges used for basis-array sizing.
    int mnyq2;
    int nnyq2;

   private:
    void compute_derived_sizes();
};

}  // namespace vfield

#endif  // VFIELD_COMMON_SIZES_HPP_
