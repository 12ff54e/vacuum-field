// test_float_smoke.cu — float pipeline vs the library's own double run.
//
// Runs the full cth-like case in float and compares the outputs against a
// double run of the same pipeline (not against the Fortran goldens): the
// pointwise arrays at 1e-4 relative-absolute and the accumulated quantities
// (potential coefficients, bsqvac — reduced sums) at 1e-3. The dense solve
// is double in both builds, so the float leg isolates the float-precision
// propagation through the geometry/integrals/analysis.
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
using vfield::test::flatArray;
using vfield::test::loadGolden;
using vfield::test::max_rel_diff;
using vfield::test::summary;
using vfield::test::toDevice;
using vfield::test::toHost;

namespace {

const std::string DATA_DIR = "tests/data/cth_like_free_bdy/";

// Runs one iteration (full update) at precision T and returns the outputs.
template <class T>
void runCase(int iter,
             const Sizes& sizes,
             const FourierBasis&,
             const std::vector<double>& rcc,
             const std::vector<double>& rss,
             const std::vector<double>& zsc,
             const std::vector<double>& zcs,
             const std::vector<double>& raxis,
             const std::vector<double>& zaxis,
             double net_toroidal_current,
             int sign_j,
             std::vector<T>* potu,
             std::vector<T>* potv,
             std::vector<T>* bsubu,
             std::vector<T>* bsubv,
             std::vector<T>* bsqvac,
             std::vector<T>* pot) {
    typename VacuumFieldSolver<T>::Params params(sizes);
    params.coil_currents = {4700.0, 1000.0};
    params.mgrid_file = DATA_DIR + "../mgrid_cth_like.nc";
    VacuumFieldSolver<T> solver(params);

    std::vector<T> rcc_t(rcc.begin(), rcc.end());
    std::vector<T> rss_t(rss.begin(), rss.end());
    std::vector<T> zsc_t(zsc.begin(), zsc.end());
    std::vector<T> zcs_t(zcs.begin(), zcs.end());
    std::vector<T> raxis_t(raxis.begin(), raxis.end());
    std::vector<T> zaxis_t(zaxis.begin(), zaxis.end());
    auto d_rcc = toDevice(rcc_t);
    auto d_rss = toDevice(rss_t);
    auto d_zsc = toDevice(zsc_t);
    auto d_zcs = toDevice(zcs_t);
    auto d_raxis = toDevice(raxis_t);
    auto d_zaxis = toDevice(zaxis_t);

    T bsubu_vac = 0, bsubv_vac = 0;
    solver.update(d_rcc.data(), d_rss.data(), nullptr, nullptr, d_zsc.data(),
                  d_zcs.data(), nullptr, nullptr, sign_j, d_raxis.data(),
                  d_zaxis.data(), &bsubu_vac, &bsubv_vac,
                  static_cast<T>(net_toroidal_current), true);

    *potu = toHost(solver.potU(), sizes.nZnT);
    *potv = toHost(solver.potV(), sizes.nZnT);
    *bsubu = toHost(solver.bSubU(), sizes.nZnT);
    *bsubv = toHost(solver.bSubV(), sizes.nZnT);
    *bsqvac = toHost(solver.bSqVac(), sizes.nZnT);
    const int mnpd = (2 * sizes.ntor + 1) * (sizes.mpol + 2);
    *pot = toHost(solver.potential(), mnpd);
    (void)iter;
}

}  // namespace

int main() {
    const json::Value vacuum = loadGolden(
        DATA_DIR +
        "vac1n_vacuum/vac1n_vacuum_00015_000053_01.cth_like_free_bdy.json");
    const int sign_j =
        static_cast<int>(static_cast<double>(vacuum.at("signgs")));

    Sizes sizes(false, 5, 5, 4, 16, 36);
    FourierBasis fb(sizes);
    std::vector<double> rcc(sizes.mnsize), rss(sizes.mnsize);
    fb.cosToCcSs(flatArray(vacuum.at("rmnc")), rcc, rss, sizes.ntor,
                 sizes.mpol);
    std::vector<double> zsc(sizes.mnsize), zcs(sizes.mnsize);
    fb.sinToScCs(flatArray(vacuum.at("zmns")), zsc, zcs, sizes.ntor,
                 sizes.mpol);
    const std::vector<double> raxis = flatArray(vacuum.at("raxis_nestor"));
    const std::vector<double> zaxis = flatArray(vacuum.at("zaxis_nestor"));
    const double net_toroidal_current =
        static_cast<double>(vacuum.at("plascur")) /
        (4.0 * std::numbers::pi * 1.0e-7);

    std::vector<double> potu_d, potv_d, bsubu_d, bsubv_d, bsqvac_d, pot_d;
    runCase<double>(53, sizes, fb, rcc, rss, zsc, zcs, raxis, zaxis,
                    net_toroidal_current, sign_j, &potu_d, &potv_d, &bsubu_d,
                    &bsubv_d, &bsqvac_d, &pot_d);
    std::vector<float> potu_f, potv_f, bsubu_f, bsubv_f, bsqvac_f, pot_f;
    runCase<float>(53, sizes, fb, rcc, rss, zsc, zcs, raxis, zaxis,
                   net_toroidal_current, sign_j, &potu_f, &potv_f, &bsubu_f,
                   &bsubv_f, &bsqvac_f, &pot_f);

    check(max_rel_diff(potu_f, potu_d) < 1e-4, "potu (float vs double)");
    check(max_rel_diff(potv_f, potv_d) < 1e-4, "potv (float vs double)");
    check(max_rel_diff(bsubu_f, bsubu_d) < 1e-4, "bsubu (float vs double)");
    check(max_rel_diff(bsubv_f, bsubv_d) < 1e-4, "bsubv (float vs double)");
    check(max_rel_diff(bsqvac_f, bsqvac_d) < 1e-3,
          "bsqvac (float vs double, accumulated)");
    check(max_rel_diff(pot_f, pot_d) < 1e-3,
          "potential coefficients (float vs double)");

    return summary();
}
