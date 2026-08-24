// test_laplace_solver.cu — manufactured single-mode round trip.
//
// Port of vmecpp's TransformGreensFunctionDerivativeTest: inject a greenp
// that is exactly one Fourier mode and verify the analysis lands in the
// right grpmn slot with the expected 0.5 amplitude (the scale factors of the
// integration-weighted basis cancel). Parameterised over lasym: the
// symmetric path injects sin(m0*theta - n0*phi) (odd -> grpmn_sin), the
// lasym path injects cos(m0*theta - n0*phi) (even -> grpmn_cos).
#include "vfield/common/fourier_basis.hpp"
#include "vfield/common/fourier_basis_device.cuh"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/laplace_solver_operator.hpp"
#include "vfield_test_cuda_helper.cuh"

#include <cmath>
#include <numbers>
#include <vector>

using vfield::FourierBasis;
using vfield::FourierBasisDevice;
using vfield::LaplaceSolverOperator;
using vfield::Sizes;
using vfield::test::check;
using vfield::test::is_close_rel_abs;
using vfield::test::summary;
using vfield::test::to_device;
using vfield::test::to_host;

namespace {

template <class T>
void run_case(bool lasym, int m0, int n0, double tol) {
    const int nfp = 5;
    const int mpol = 4;
    const int ntor = 4;
    const int ntheta = 0;              // auto-adjusted by Sizes
    const int nzeta = 4 * (ntor + 1);  // must satisfy nzeta >= 2*ntor+4

    Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);
    FourierBasis fb(sizes);
    FourierBasisDevice<T> fbd(fb, sizes.lasym, sizes.nThetaEven);
    LaplaceSolverOperator<T> ls(sizes, fbd);

    const int nf = ntor;
    const int mf = mpol + 1;
    const int mnpd = (2 * nf + 1) * (mf + 1);

    // Build greenp[klpRel * nThetaEven * nZeta + l * nZeta + k]: for each
    // source point, inject exactly one mode.
    const int nThetaEven = sizes.nThetaEven;
    const int nZeta = sizes.nZeta;
    std::vector<T> greenp(sizes.nZnT * nThetaEven * nZeta, T(0));
    for (int klpRel = 0; klpRel < sizes.nZnT; ++klpRel) {
        for (int l = 0; l < nThetaEven; ++l) {
            const double theta = 2.0 * std::numbers::pi * l / nThetaEven;
            for (int k = 0; k < nZeta; ++k) {
                const double phi = 2.0 * std::numbers::pi * k / nZeta;
                const int idx = klpRel * nThetaEven * nZeta + l * nZeta + k;
                greenp[idx] =
                    static_cast<T>(lasym ? std::cos(m0 * theta - n0 * phi)
                                         : std::sin(m0 * theta - n0 * phi));
            }
        }
    }
    auto d_greenp = to_device(greenp);
    ls.transform_greens_function_derivative(d_greenp.data());

    // Expected: the basis normalization makes the trapezoidal DFT of sin^2 /
    // cos^2 yield 0.5 per mode at the injected (m0, +n0) slot.
    const int idx_posn = (nf + n0) * (mf + 1) + m0;
    constexpr double EXPECTED = 0.5;

    bool ok = true;
    if (!lasym) {
        const auto grpmn = to_host(ls.grpmn_sin(), mnpd * sizes.nZnT);
        for (int klpRel = 0; klpRel < sizes.nZnT; ++klpRel) {
            if (!is_close_rel_abs(
                    EXPECTED,
                    static_cast<double>(grpmn[idx_posn * sizes.nZnT + klpRel]),
                    tol)) {
                ok = false;
            }
        }
        // Every other slot must be ~0 — except, for m0 == 0, the -n0 slot:
        // the m=0 basis functions of +-n differ only by a sign, so the
        // injected mode also has amplitude -0.5 there (the gauge
        // duplication BuildMatrix removes later).
        for (int mn = 0; mn < mnpd; ++mn) {
            if (mn == idx_posn) continue;
            if (m0 == 0 && mn == (nf - n0) * (mf + 1)) continue;
            for (int klpRel = 0; klpRel < sizes.nZnT; ++klpRel) {
                if (std::fabs(static_cast<double>(
                        grpmn[mn * sizes.nZnT + klpRel])) > tol) {
                    ok = false;
                }
            }
        }
    } else {
        const auto grpmn = to_host(ls.grpmn_cos(), mnpd * sizes.nZnT);
        for (int klpRel = 0; klpRel < sizes.nZnT; ++klpRel) {
            if (!is_close_rel_abs(
                    EXPECTED,
                    static_cast<double>(grpmn[idx_posn * sizes.nZnT + klpRel]),
                    tol)) {
                ok = false;
            }
        }
    }

    const std::string label = std::string(lasym ? "lasym" : "symm") +
                              " round trip (m0=" + std::to_string(m0) +
                              ", n0=" + std::to_string(n0) + ")";
    check(ok, label);
}

}  // namespace

int main() {
    run_case<double>(false, 1, 1, 1e-12);
    run_case<double>(false, 3, 4, 1e-12);
    run_case<double>(false, 0, 2, 1e-12);
    run_case<double>(true, 2, 3, 1e-12);
    run_case<float>(false, 1, 1, 1e-4);
    return summary();
}
