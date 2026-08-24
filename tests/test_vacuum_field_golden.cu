// test_vacuum_field_golden.cu — vac1n_bsqvac comparison (double).
//
// Golden gate, end to end: the VacuumFieldSolver driver runs the complete
// NESTOR pipeline on the cth-like inputs and the outputs (potential
// coefficients, potu/potv, bsubu/bsubv, bsqvac, brv/bphiv/bzv) are compared
// elementwise against the Fortran-VMEC vac1n_bsqvac dumps at 1e-10 (vmecpp's
// own tolerance). The solver state is carried across iterations 53 (full
// update) and 54 (partial update with the stale Laplace factor), exactly
// like vmecpp's single-Vmec flow.
#include "golden_data.hpp"
#include "vfield/common/fourier_basis.hpp"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/vacuum_field_solver.hpp"

#include <iomanip>
#include <numbers>
#include <sstream>
#include <string>

using vfield::FourierBasis;
using vfield::Sizes;
using vfield::VacuumFieldSolver;
using vfield::test::check;
using vfield::test::compare_zeta_fast;
using vfield::test::flat_array;
using vfield::test::is_close_rel_abs;
using vfield::test::load_golden;
using vfield::test::nested_array;
using vfield::test::summary;
using vfield::test::to_device;
using vfield::test::to_host;

namespace {

const std::string DATA_DIR = "tests/data/cth_like_free_bdy/";

std::string json_path(const std::string& checkpoint, int iter) {
    std::ostringstream oss;
    oss << DATA_DIR << checkpoint << "/" << checkpoint << "_00015_"
        << std::setw(6) << std::setfill('0') << iter
        << "_01.cth_like_free_bdy.json";
    return oss.str();
}

void run_iteration(int iter,
                  VacuumFieldSolver<double>& solver,
                  const Sizes& sizes) {
    const json::Value vacuum = load_golden(json_path("vac1n_vacuum", iter));
    const json::Value bsqvac = load_golden(json_path("vac1n_bsqvac", iter));

    const bool full_update =
        (static_cast<double>(vacuum.at("ivac_skip")) == 0.0);
    const int sign_j =
        static_cast<int>(static_cast<double>(vacuum.at("signgs")));

    FourierBasis fb(sizes);
    std::vector<double> rcc(sizes.mnsize), rss(sizes.mnsize);
    fb.cos_to_cc_ss(flat_array(vacuum.at("rmnc")), rcc, rss, sizes.ntor,
                 sizes.mpol);
    std::vector<double> zsc(sizes.mnsize), zcs(sizes.mnsize);
    fb.sin_to_sc_cs(flat_array(vacuum.at("zmns")), zsc, zcs, sizes.ntor,
                 sizes.mpol);

    auto d_rcc = to_device(rcc);
    auto d_rss = to_device(rss);
    auto d_zsc = to_device(zsc);
    auto d_zcs = to_device(zcs);
    auto d_raxis = to_device(flat_array(vacuum.at("raxis_nestor")));
    auto d_zaxis = to_device(flat_array(vacuum.at("zaxis_nestor")));

    const double net_toroidal_current =
        static_cast<double>(vacuum.at("plascur")) /
        (4.0 * std::numbers::pi * 1.0e-7);

    double bsubu_vac = 0.0, bsubv_vac = 0.0;
    solver.update(d_rcc.data(), d_rss.data(), nullptr, nullptr, d_zsc.data(),
                  d_zcs.data(), nullptr, nullptr, sign_j, d_raxis.data(),
                  d_zaxis.data(), bsubu_vac, bsubv_vac, net_toroidal_current,
                  full_update);
    (void)bsubu_vac;
    (void)bsubv_vac;

    const double tol = 1e-10;
    const std::string iter_name = "iter " + std::to_string(iter);

    // Potential coefficients (potsin, flat [mnpd]).
    const int mnpd = (2 * sizes.ntor + 1) * (sizes.mpol + 2);
    const auto pot = to_host(solver.potential(), mnpd);
    bool pot_ok = true;
    for (int mn = 0; mn < mnpd; ++mn) {
        if (!is_close_rel_abs(static_cast<double>(bsqvac.at("potsin")[mn]),
                              pot[mn], tol)) {
            pot_ok = false;
        }
    }
    check(pot_ok, ("potsin " + iter_name));

    check(compare_zeta_fast(nested_array(bsqvac.at("potu")),
                          to_host(solver.pot_u(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("potu " + iter_name));
    check(compare_zeta_fast(nested_array(bsqvac.at("potv")),
                          to_host(solver.pot_v(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("potv " + iter_name));
    check(compare_zeta_fast(nested_array(bsqvac.at("bsubu")),
                          to_host(solver.b_sub_u(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("bsubu " + iter_name));
    check(compare_zeta_fast(nested_array(bsqvac.at("bsubv")),
                          to_host(solver.b_sub_v(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("bsubv " + iter_name));
    check(compare_zeta_fast(nested_array(bsqvac.at("bsqvac")),
                          to_host(solver.b_sq_vac(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("bsqvac " + iter_name));
    check(compare_zeta_fast(nested_array(bsqvac.at("brv")),
                          to_host(solver.vacuum_b_r(), sizes.nZnT),
                          sizes.nThetaEff, tol) == 0,
          ("brv " + iter_name));
    check(compare_zeta_fast(nested_array(bsqvac.at("bphiv")),
                          to_host(solver.vacuum_b_phi(), sizes.nZnT),
                          sizes.nThetaEff, tol) == 0,
          ("bphiv " + iter_name));
    check(compare_zeta_fast(nested_array(bsqvac.at("bzv")),
                          to_host(solver.vacuum_b_z(), sizes.nZnT),
                          sizes.nThetaEff, tol) == 0,
          ("bzv " + iter_name));
}

}  // namespace

int main() {
    Sizes sizes(false, 5, 5, 4, 16, 36);
    VacuumFieldSolver<double>::Params params(sizes);
    params.coil_currents = {4700.0, 1000.0};
    params.mgrid_file = DATA_DIR + "../mgrid_cth_like.nc";
    VacuumFieldSolver<double> solver(params);

    run_iteration(53, solver, sizes);
    run_iteration(54, solver, sizes);
    return summary();
}
