// test_singular_coefficients.cpp — cmn/cmns expansion coefficient checks.
//
// Host-only. Verifies cmn against the closed-form factorial expression
// (vmecpp's CheckConstants) and cmns against the Fortran vac1n_precal dump
// (which carries the alp = 2*pi/nfp factor and stores the [l][m][n]
// transposed layout).
#include "vfield/JsonParser.h"
#include "vfield/common/singular_coefficients.hpp"
#include "vfield_test.h"

#include <cmath>
#include <numbers>
#include <string>

using vfield::SingularCoefficients;
using vfield::test::check;
using vfield::test::is_close_rel_abs;
using vfield::test::summary;

int main() {
    // Factorial-formula check over a range of resolutions.
    {
        const int nf = 6;
        const int mf = 7;
        SingularCoefficients sc(nf, mf);

        bool ok = true;
        for (int n = 0; n < nf + 1; ++n) {
            for (int m = 0; m < mf + 1; ++m) {
                for (int l = std::abs(m - n); l <= m + n; l += 2) {
                    const int lnm = (l * (nf + 1) + n) * (mf + 1) + m;

                    const int sign = ((l - m + n) / 2) % 2 == 0 ? 1 : -1;

                    // need to compute n! / m!
                    // Note: n! = gamma(n + 1)
                    // Note: lgamma(n + 1) == log(n!)
                    // exp(lgamma(n +1) - lgamma(m+1)) == n! / m!
                    const double num_fac = std::lgamma((m + n + l) / 2 + 1);
                    const double den_fac_1 = std::lgamma((m + n - l) / 2 + 1);
                    const double den_fac_2 =
                        std::lgamma((l + std::abs(m - n)) / 2 + 1);
                    const double den_fac_3 =
                        std::lgamma((l - std::abs(m - n)) / 2 + 1);

                    const double cmn_ref =
                        sign * std::exp(num_fac - den_fac_1 - den_fac_2 - den_fac_3);

                    if (!is_close_rel_abs(cmn_ref, sc.cmn[lnm], 1e-12)) {
                        ok = false;
                    }
                }  // l
            }  // m
        }  // n
        check(ok, "cmn matches the factorial formula");
    }

    // Golden cmns check (cth-like: nfp=5, nf=4, mf=6).
    {
        const std::string path =
            "tests/data/cth_like_free_bdy/vac1n_precal/"
            "vac1n_precal_00015_000053_01.cth_like_free_bdy.json";
        const json::Value precal = json::parse_file(path);
        const int nf = 4;
        const int mf = 6;
        const int nfp = 5;
        SingularCoefficients sc(nf, mf);

        // In Fortran VMEC/Nestor, a factor of 2 pi / nfp (called `alp`
        // there) is included in cmns that must be accounted for here; the
        // Fortran layout is [l][m][n], the library layout is
        // ((l * (nf + 1) + n) * (mf + 1) + m).
        const double alp = 2.0 * std::numbers::pi / nfp;

        bool ok = true;
        for (int l = 0; l < 1 + nf + mf; ++l) {
            for (int m = 0; m < mf + 1; ++m) {
                for (int n = 0; n < nf + 1; ++n) {
                    const int lnm = (l * (nf + 1) + n) * (mf + 1) + m;
                    const double golden =
                        static_cast<double>(precal.at("cmns")[l][m][n]);
                    if (!is_close_rel_abs(golden, alp * sc.cmns[lnm], 1e-14)) {
                        ok = false;
                    }
                }  // n
            }  // m
        }  // l
        check(ok, "cmns matches vac1n_precal (alp factor, [l][m][n] layout)");
    }

    return summary();
}
