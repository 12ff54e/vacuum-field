// test_singular_golden.cu — vac1n_analyt comparison (double).
//
// Golden gate: runs the full surface-geometry + external-field + singular
// pipeline on the cth-like inputs and compares the T/S tables (direct), the
// bvec RHS (scaled by alp * (2 pi)^2, Fortran [m][nf+n] layout), and the
// grpmn kernel (scaled by alp, Fortran [m][nf+n][k][l] layout) against the
// Fortran-VMEC vac1n_analyt dumps at 1e-9 (vmecpp's own tolerance for this
// module). Iteration 54 has ivac_skip=1, so S/grpmn are only compared for
// iteration 53.
#include "golden_data.hpp"
#include "vfield/common/fourier_basis.hpp"
#include "vfield/common/fourier_basis_device.cuh"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/external_field_operator.hpp"
#include "vfield/free_boundary/mgrid_provider.hpp"
#include "vfield/free_boundary/singular_integrals_operator.hpp"
#include "vfield/free_boundary/surface_geometry_operator.hpp"

#include <iomanip>
#include <numbers>
#include <sstream>
#include <string>

using vfield::ExternalFieldOperator;
using vfield::FourierBasis;
using vfield::FourierBasisDevice;
using vfield::MgridProvider;
using vfield::SingularIntegralsOperator;
using vfield::Sizes;
using vfield::SurfaceGeometryOperator;
using vfield::test::check;
using vfield::test::flat_array;
using vfield::test::is_close_rel_abs;
using vfield::test::load_golden;
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

void run_iteration(int iter) {
    const json::Value vacuum = load_golden(json_path("vac1n_vacuum", iter));
    const json::Value analyt = load_golden(json_path("vac1n_analyt", iter));

    const bool full_update =
        (static_cast<double>(vacuum.at("ivac_skip")) == 0.0);
    const int sign_j =
        static_cast<int>(static_cast<double>(vacuum.at("signgs")));

    Sizes sizes(false, 5, 5, 4, 16, 36);
    FourierBasis fb(sizes);

    std::vector<double> rcc(sizes.mnsize), rss(sizes.mnsize);
    fb.cos_to_cc_ss(flat_array(vacuum.at("rmnc")), rcc, rss, sizes.ntor,
                 sizes.mpol);
    std::vector<double> zsc(sizes.mnsize), zcs(sizes.mnsize);
    fb.sin_to_sc_cs(flat_array(vacuum.at("zmns")), zsc, zcs, sizes.ntor,
                 sizes.mpol);

    MgridProvider mgrid;
    mgrid.load_file(DATA_DIR + "../mgrid_cth_like.nc",
                   std::vector<double>{4700.0, 1000.0});

    FourierBasisDevice<double> fbd(fb, sizes.lasym, sizes.nThetaEven);
    SurfaceGeometryOperator<double> sg(sizes, fbd);
    ExternalFieldOperator<double> ef(sizes, sg, mgrid);
    SingularIntegralsOperator<double> si(sizes, fbd, sg);

    const int nf = sizes.ntor;
    const int mf = sizes.mpol + 1;

    auto d_rcc = to_device(rcc);
    auto d_rss = to_device(rss);
    auto d_zsc = to_device(zsc);
    auto d_zcs = to_device(zcs);
    auto d_raxis = to_device(flat_array(vacuum.at("raxis_nestor")));
    auto d_zaxis = to_device(flat_array(vacuum.at("zaxis_nestor")));

    sg.update(d_rcc.data(), d_rss.data(), nullptr, nullptr, d_zsc.data(),
              d_zcs.data(), nullptr, nullptr, sign_j, full_update);
    const double net_toroidal_current =
        static_cast<double>(vacuum.at("plascur")) /
        (4.0 * std::numbers::pi * 1.0e-7);
    ef.update(d_raxis.data(), d_zaxis.data(), net_toroidal_current);
    si.update(ef.b_dot_n(), full_update);

    const double tol = 1e-9;
    const std::string iter_name = "iter " + std::to_string(iter);

    // T tables: [fl][k][l], compared directly.
    const auto tlp = to_host(si.tlp(), (nf + mf + 2) * sizes.nZnT);
    const auto tlm = to_host(si.tlm(), (nf + mf + 2) * sizes.nZnT);
    bool t_ok = true;
    for (int fl = 0; fl < mf + nf + 1; ++fl) {
        for (int kl = 0; kl < sizes.nZnT; ++kl) {
            const int l = kl / sizes.nZeta;
            const int k = kl % sizes.nZeta;
            const double got = static_cast<double>(tlp[fl * sizes.nZnT + kl]);
            const double golden =
                static_cast<double>(analyt.at("all_tlp")[fl][k][l]);
            if (!is_close_rel_abs(golden, got, tol)) t_ok = false;
            const double got_m = static_cast<double>(tlm[fl * sizes.nZnT + kl]);
            const double golden_m =
                static_cast<double>(analyt.at("all_tlm")[fl][k][l]);
            if (!is_close_rel_abs(golden_m, got_m, tol)) t_ok = false;
        }
    }
    check(t_ok, ("all_tlp/all_tlm " + iter_name));

    // bvec: scale alp * (2 pi)^2, Fortran layout [m][nf+n].
    const double bvec_scale = 2.0 * std::numbers::pi / sizes.nfp * 4.0 *
                              std::numbers::pi * std::numbers::pi;
    const auto bvec = to_host(si.bvec_sin(), (2 * nf + 1) * (mf + 1));
    bool bvec_ok = true;
    for (int n = 0; n < nf + 1; ++n) {
        for (int m = 0; m < mf + 1; ++m) {
            const int idx_posn = (nf + n) * (mf + 1) + m;
            const int idx_negn = (nf - n) * (mf + 1) + m;
            const double golden_posn =
                static_cast<double>(analyt.at("bvec")[m][nf + n]);
            const double golden_negn =
                static_cast<double>(analyt.at("bvec")[m][nf - n]);
            if (!is_close_rel_abs(golden_posn, bvec_scale * bvec[idx_posn],
                                  tol)) {
                bvec_ok = false;
            }
            if (!is_close_rel_abs(golden_negn, bvec_scale * bvec[idx_negn],
                                  tol)) {
                bvec_ok = false;
            }
        }
    }
    check(bvec_ok, ("bvec " + iter_name));

    if (full_update) {
        // S tables: [fl][k][l], compared directly.
        const auto slp = to_host(si.slp(), (nf + mf + 1) * sizes.nZnT);
        const auto slm = to_host(si.slm(), (nf + mf + 1) * sizes.nZnT);
        bool s_ok = true;
        for (int fl = 0; fl < mf + nf + 1; ++fl) {
            for (int kl = 0; kl < sizes.nZnT; ++kl) {
                const int l = kl / sizes.nZeta;
                const int k = kl % sizes.nZeta;
                const double got =
                    static_cast<double>(slp[fl * sizes.nZnT + kl]);
                const double golden =
                    static_cast<double>(analyt.at("all_slp")[fl][k][l]);
                if (!is_close_rel_abs(golden, got, tol)) s_ok = false;
                const double got_m =
                    static_cast<double>(slm[fl * sizes.nZnT + kl]);
                const double golden_m =
                    static_cast<double>(analyt.at("all_slm")[fl][k][l]);
                if (!is_close_rel_abs(golden_m, got_m, tol)) s_ok = false;
            }
        }
        check(s_ok, ("all_slp/all_slm " + iter_name));

        // grpmn: scale alp, Fortran layout [m][nf+n][k][l].
        const double grpmn_scale = 2.0 * std::numbers::pi / sizes.nfp;
        const auto grpmn =
            to_host(si.grpmn_sin(), (2 * nf + 1) * (mf + 1) * sizes.nZnT);
        bool grpmn_ok = true;
        for (int n = 0; n < nf + 1; ++n) {
            for (int m = 0; m < mf + 1; ++m) {
                const int idx_posn = (nf + n) * (mf + 1) + m;
                const int idx_negn = (nf - n) * (mf + 1) + m;
                for (int kl = 0; kl < sizes.nZnT; ++kl) {
                    const int l = kl / sizes.nZeta;
                    const int k = kl % sizes.nZeta;
                    const double golden_posn = static_cast<double>(
                        analyt.at("grpmn")[m][nf + n][k][l]);
                    if (!is_close_rel_abs(
                            golden_posn,
                            grpmn_scale * grpmn[idx_posn * sizes.nZnT + kl],
                            tol)) {
                        grpmn_ok = false;
                    }
                    const double golden_negn = static_cast<double>(
                        analyt.at("grpmn")[m][nf - n][k][l]);
                    if (!is_close_rel_abs(
                            golden_negn,
                            grpmn_scale * grpmn[idx_negn * sizes.nZnT + kl],
                            tol)) {
                        grpmn_ok = false;
                    }
                }
            }
        }
        check(grpmn_ok, ("grpmn " + iter_name));
    }
}

}  // namespace

int main() {
    run_iteration(53);
    run_iteration(54);
    return summary();
}
