// test_sizes.cpp — Sizes derived-count and weight checks.
//
// Host-only. Locks the vmecpp count adjustments (Nyquist raisings, reduced
// grid, trapezoidal weights) with the cth-like free-boundary case as the
// reference geometry.
#include "vfield/common/sizes.hpp"
#include "vfield_test.h"

#include <cmath>
#include <stdexcept>

using vfield::Sizes;
using vfield::test::check;
using vfield::test::expect_near;
using vfield::test::summary;

int main() {
    // cth_like_free_bdy: nfp=5, mpol=5, ntor=4, ntheta=16, nzeta=36.
    Sizes cth(false, 5, 5, 4, 16, 36);
    check(cth.ntheta == 16, "cth: ntheta stays 16 (>= 2*mpol+6)");
    check(cth.nZeta == 36, "cth: nzeta stays 36 (>= 2*ntor+4)");
    check(cth.nThetaEven == 16, "cth: nThetaEven == 16");
    check(cth.nThetaReduced == 9, "cth: nThetaReduced == 9");
    check(cth.nThetaEff == 9, "cth: symmetric -> nThetaEff == reduced");
    check(cth.nZnT == 324, "cth: nZnT == 36*9");
    check(cth.mnsize == 25, "cth: mnsize == mpol*(ntor+1)");
    check(cth.mnmax == 41, "cth: mnmax == (ntor+1)+(mpol-1)*(2*ntor+1)");
    check(cth.lthreed, "cth: lthreed");
    check(cth.mnyq2 == 16, "cth: mnyq2 == 2*nThetaEven/2");
    check(cth.nnyq2 == 36, "cth: nnyq2 == 2*nZeta/2");

    // Trapezoidal weights: endpoints halved in the symmetric case; the sum
    // over the reduced grid equals the full-interval normalization 1/nZeta.
    double wsum = 0.0;
    for (double w : cth.wInt) wsum += w;
    expect_near(wsum, 1.0 / 36.0, 1e-15, "cth: sum(wInt) == 1/nZeta");
    expect_near(cth.wInt[0], 1.0 / (36.0 * 8.0 * 2.0), 1e-18,
                "cth: wInt endpoint halved");

    // Nyquist raising: ntheta below the floor is raised.
    Sizes small(false, 3, 2, 1, 2, 4);
    check(small.ntheta == 10, "small: ntheta raised to 2*mpol+6");
    check(small.nZeta == 6, "small: nzeta raised to 2*ntor+4");

    // Axisymmetric: ntor == 0 forces nzeta == 1 (solovev_free_bdy shape).
    Sizes axi(false, 1, 4, 0, 24, 1);
    check(axi.nZeta == 1, "axisym: nzeta == 1");
    check(!axi.lthreed, "axisym: not lthreed");
    check(axi.nZnT == axi.nThetaReduced, "axisym: nZnT == nThetaReduced");
    double awsum = 0.0;
    for (double w : axi.wInt) awsum += w;
    expect_near(awsum, 1.0, 1e-15, "axisym: sum(wInt) == 1 (nZeta == 1)");

    // lasym: full poloidal range, uniform weights.
    Sizes asym(true, 5, 5, 4, 16, 36);
    check(asym.nThetaEff == 16, "lasym: nThetaEff == nThetaEven");
    check(asym.nZnT == 36 * 16, "lasym: nZnT == nZeta*nThetaEven");
    double lsum = 0.0;
    for (double w : asym.wInt) lsum += w;
    expect_near(lsum, 1.0 / 36.0, 1e-15, "lasym: sum(wInt) == 1/nZeta");
    expect_near(asym.wInt[0], 1.0 / (36.0 * 16.0), 1e-18,
                "lasym: uniform weights, no endpoint halving");

    // Invalid inputs throw.
    bool threw = false;
    try {
        Sizes bad(false, 0, 4, 0, 24, 1);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "nfp < 1 throws");

    return summary();
}
