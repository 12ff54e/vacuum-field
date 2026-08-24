// test_lasym_parity.cu — lasym with zero antisymmetric coefficients ==
// symmetric result.
//
// The lasym pipeline on a stellarator-symmetric boundary with all
// antisymmetric coefficients set to zero must reproduce the symmetric
// pipeline on the reduced poloidal range (the second half of the lasym run
// is the odd mirror of the first, so the Fourier sums agree). The external
// field is driven by the axis current (physically symmetric), so the lasym
// pipeline sees the exact odd mirror of the symmetric integrand on the
// second poloidal half. No Fortran golden exists for lasym — this is the
// equivalence gate.
#include "vfield/common/fourier_basis.hpp"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/vacuum_field_solver.hpp"
#include "vfield_test_cuda_helper.cuh"

#include <cmath>
#include <numbers>
#include <vector>

using vfield::FourierBasis;
using vfield::Sizes;
using vfield::VacuumFieldSolver;
using vfield::test::check;
using vfield::test::max_rel_diff;
using vfield::test::summary;
using vfield::test::to_device;
using vfield::test::to_host;

namespace {

constexpr int NFP = 5;
constexpr int MPOL = 5;
constexpr int NTOR = 4;
constexpr int NTHETA = 16;
constexpr int NZETA = 36;
constexpr int SIGN_J = -1;
// The axis-current drive (A); any nonzero value works.
constexpr double AXIS_CURRENT = 1.0e6;

void make_coefficients(std::vector<double>* rcc,
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

struct Outputs {
    std::vector<double> potu, potv, bsubu, bsubv, bsqvac, pot;
};

// Runs the pipeline with the axis-current external field and returns the
// outputs over the REDUCED poloidal range.
Outputs run_pipeline(const Sizes& sizes,
                    const std::vector<double>& rcc,
                    const std::vector<double>& rss,
                    const std::vector<double>& zsc,
                    const std::vector<double>& zcs,
                    const std::vector<double>& raxis,
                    const std::vector<double>& zaxis) {
    // Zero fixed field: the external field comes from the axis current only.
    std::vector<double> fixed_br(sizes.nZnT, 0.0);
    std::vector<double> fixed_bp(sizes.nZnT, 0.0);
    std::vector<double> fixed_bz(sizes.nZnT, 0.0);

    VacuumFieldSolver<double>::Params params(sizes);
    params.fixed_br = fixed_br;
    params.fixed_bp = fixed_bp;
    params.fixed_bz = fixed_bz;
    VacuumFieldSolver<double> solver(params);

    std::vector<double> zero(sizes.mnsize, 0.0);
    auto d_rcc = to_device(rcc);
    auto d_rss = to_device(rss);
    auto d_zsc = to_device(zsc);
    auto d_zcs = to_device(zcs);
    auto d_zero = to_device(zero);
    auto d_raxis = to_device(raxis);
    auto d_zaxis = to_device(zaxis);

    double bsubu_vac = 0, bsubv_vac = 0;
    if (sizes.lasym) {
        solver.update(d_rcc.data(), d_rss.data(), d_zero.data(), d_zero.data(),
                      d_zsc.data(), d_zcs.data(), d_zero.data(), d_zero.data(),
                      SIGN_J, d_raxis.data(), d_zaxis.data(), &bsubu_vac,
                      &bsubv_vac, AXIS_CURRENT, true);
    } else {
        solver.update(d_rcc.data(), d_rss.data(), nullptr, nullptr,
                      d_zsc.data(), d_zcs.data(), nullptr, nullptr, SIGN_J,
                      d_raxis.data(), d_zaxis.data(), &bsubu_vac, &bsubv_vac,
                      AXIS_CURRENT, true);
    }

    Outputs out;
    out.potu = to_host(solver.pot_u(), sizes.nZnT);
    out.potv = to_host(solver.pot_v(), sizes.nZnT);
    out.bsubu = to_host(solver.b_sub_u(), sizes.nZnT);
    out.bsubv = to_host(solver.b_sub_v(), sizes.nZnT);
    out.bsqvac = to_host(solver.b_sq_vac(), sizes.nZnT);
    const int mnpd = (2 * sizes.ntor + 1) * (sizes.mpol + 2);
    out.pot = to_host(solver.potential(), mnpd);
    // keep only the reduced range (the lasym outputs span the full grid)
    out.potu.resize(sizes.nZeta * sizes.nThetaReduced);
    out.potv.resize(sizes.nZeta * sizes.nThetaReduced);
    out.bsubu.resize(sizes.nZeta * sizes.nThetaReduced);
    out.bsubv.resize(sizes.nZeta * sizes.nThetaReduced);
    out.bsqvac.resize(sizes.nZeta * sizes.nThetaReduced);
    return out;
}

}  // namespace

int main() {
    std::vector<double> rcc, rss, zsc, zcs;
    make_coefficients(&rcc, &rss, &zsc, &zcs);
    std::vector<double> raxis(NZETA), zaxis(NZETA);
    for (int k = 0; k < NZETA; ++k) {
        raxis[k] = 0.6 + 0.02 * std::cos(2.0 * std::numbers::pi * k / NZETA);
        zaxis[k] = 0.1 * std::sin(2.0 * std::numbers::pi * k / NZETA);
    }

    Sizes sizes_sym(false, NFP, MPOL, NTOR, NTHETA, NZETA);
    const Outputs sym =
        run_pipeline(sizes_sym, rcc, rss, zsc, zcs, raxis, zaxis);

    Sizes sizes_asym(true, NFP, MPOL, NTOR, NTHETA, NZETA);
    const Outputs asym =
        run_pipeline(sizes_asym, rcc, rss, zsc, zcs, raxis, zaxis);

    const double tol = 1e-12;
    check(max_rel_diff(asym.potu, sym.potu) < tol, "potu parity");
    check(max_rel_diff(asym.potv, sym.potv) < tol, "potv parity");
    check(max_rel_diff(asym.bsubu, sym.bsubu) < tol, "bsubu parity");
    check(max_rel_diff(asym.bsubv, sym.bsubv) < tol, "bsubv parity");
    check(max_rel_diff(asym.bsqvac, sym.bsqvac) < tol, "bsqvac parity");
    check(max_rel_diff(asym.pot, sym.pot) < tol, "potential parity");

    return summary();
}
