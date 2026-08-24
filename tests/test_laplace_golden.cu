// test_laplace_golden.cu — vac1n_fourp/fouri/solver comparison (double).
//
// Golden gate: runs the whole pipeline up to the Laplace solve on the
// cth-like inputs and compares, with vmecpp's own scale factors:
//   vac1n_fourp:  grpmn (regularized part, scale 2*pi; the singular part at
//                 scale 2*pi/nfp is subtracted from the Fortran total, and
//                 the post-accumulation total is also compared directly);
//   vac1n_fouri:  source (2*pi), bcos/bsin (2*pi), actemp/astemp (2*pi),
//                 bvec (2*pi * (2*pi)^2), amatrix (2*pi * (2*pi)^2);
//   vac1n_solver: potvac_in (2*pi * (2*pi)^2), amatrix (2*pi * (2*pi)^2),
//                 potvac_out (direct — cross-validates the host LU against
//                 Fortran LAPACK).
// All at 1e-9. The fourp/fouri/solver dumps exist only for iteration 53
// (full updates), the fouri comparisons for both iterations where the dump
// exists.
#include "golden_data.hpp"
#include "vfield/common/fourier_basis.hpp"
#include "vfield/common/fourier_basis_device.cuh"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/external_field_operator.hpp"
#include "vfield/free_boundary/laplace_solver_operator.hpp"
#include "vfield/free_boundary/mgrid_provider.hpp"
#include "vfield/free_boundary/regularized_integrals_operator.hpp"
#include "vfield/free_boundary/singular_integrals_operator.hpp"
#include "vfield/free_boundary/surface_geometry_operator.hpp"

#include <iomanip>
#include <numbers>
#include <sstream>
#include <string>

using vfield::ExternalFieldOperator;
using vfield::FourierBasis;
using vfield::FourierBasisDevice;
using vfield::LaplaceSolverOperator;
using vfield::MgridProvider;
using vfield::RegularizedIntegralsOperator;
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

// Runs the FULL pipeline of the given iteration and performs the
// fourp/fouri/solver comparisons (the dumps only exist for full updates).
void run_full_update(int iter,
                   const std::vector<double>& rcc,
                   const std::vector<double>& rss,
                   const std::vector<double>& zsc,
                   const std::vector<double>& zcs,
                   const std::vector<double>& raxis,
                   const std::vector<double>& zaxis,
                   SurfaceGeometryOperator<double>& sg,
                   ExternalFieldOperator<double>& ef,
                   SingularIntegralsOperator<double>& si,
                   RegularizedIntegralsOperator<double>& ri,
                   LaplaceSolverOperator<double>& ls,
                   const std::vector<double>& extcur) {
    const json::Value vacuum = load_golden(json_path("vac1n_vacuum", iter));
    (void)extcur;

    Sizes sizes(false, 5, 5, 4, 16, 36);
    const int nf = sizes.ntor;
    const int mf = sizes.mpol + 1;
    const int mnpd = (2 * nf + 1) * (mf + 1);

    const bool full_update =
        (static_cast<double>(vacuum.at("ivac_skip")) == 0.0);
    const int sign_j =
        static_cast<int>(static_cast<double>(vacuum.at("signgs")));

    auto d_rcc = to_device(rcc);
    auto d_rss = to_device(rss);
    auto d_zsc = to_device(zsc);
    auto d_zcs = to_device(zcs);
    auto d_raxis = to_device(raxis);
    auto d_zaxis = to_device(zaxis);

    sg.update(d_rcc.data(), d_rss.data(), nullptr, nullptr, d_zsc.data(),
              d_zcs.data(), nullptr, nullptr, sign_j, full_update);
    const double net_toroidal_current =
        static_cast<double>(vacuum.at("plascur")) /
        (4.0 * std::numbers::pi * 1.0e-7);
    ef.update(d_raxis.data(), d_zaxis.data(), net_toroidal_current);
    si.update(ef.b_dot_n(), full_update);

    if (full_update) {
        ri.update(ef.b_dot_n());
        ls.transform_greens_function_derivative(ri.greenp());

        // vac1n_fourp: grpmn regularized part (2*pi scale) vs
        // Fortran total minus the singular part (2*pi/nfp scale).
        {
            const json::Value fourp = load_golden(json_path("vac1n_fourp", iter));
            const double tol = 1e-9;
            const double scale_regular = 2.0 * std::numbers::pi;
            const double scale_singular = 2.0 * std::numbers::pi / sizes.nfp;
            const auto grpmn = to_host(ls.grpmn_sin(), mnpd * sizes.nZnT);
            const auto singular = to_host(si.grpmn_sin(), mnpd * sizes.nZnT);
            bool ok = true;
            for (int n = 0; n < nf + 1; ++n) {
                for (int m = 0; m < mf + 1; ++m) {
                    const int idx_posn = (nf + n) * (mf + 1) + m;
                    const int idx_negn = (nf - n) * (mf + 1) + m;
                    for (int kl = 0; kl < sizes.nZnT; ++kl) {
                        const int l = kl / sizes.nZeta;
                        const int k = kl % sizes.nZeta;
                        const double ref_posn =
                            static_cast<double>(
                                fourp.at("grpmn")[m][nf + n][k][l]) -
                            scale_singular *
                                singular[idx_posn * sizes.nZnT + kl];
                        if (!is_close_rel_abs(
                                ref_posn,
                                scale_regular *
                                    grpmn[idx_posn * sizes.nZnT + kl],
                                tol)) {
                            ok = false;
                        }
                        const double ref_negn =
                            static_cast<double>(
                                fourp.at("grpmn")[m][nf - n][k][l]) -
                            scale_singular *
                                singular[idx_negn * sizes.nZnT + kl];
                        if (!is_close_rel_abs(
                                ref_negn,
                                scale_regular *
                                    grpmn[idx_negn * sizes.nZnT + kl],
                                tol)) {
                            ok = false;
                        }
                    }
                }
            }
            check(ok,
                  ("fourp grpmn (regularized) iter " + std::to_string(iter)));
        }

        ls.symmetrise_source_term(ri.gstore());
        ls.accumulate_full_grpmn(si.grpmn_sin(), si.grpmn_cos());

        // Post-accumulation total vs fourp directly (2*pi scale).
        {
            const json::Value fourp = load_golden(json_path("vac1n_fourp", iter));
            const double tol = 1e-9;
            const double scale = 2.0 * std::numbers::pi;
            const auto grpmn = to_host(ls.grpmn_sin(), mnpd * sizes.nZnT);
            bool ok = true;
            for (int n = 0; n < nf + 1; ++n) {
                for (int m = 0; m < mf + 1; ++m) {
                    const int idx_posn = (nf + n) * (mf + 1) + m;
                    const int idx_negn = (nf - n) * (mf + 1) + m;
                    for (int kl = 0; kl < sizes.nZnT; ++kl) {
                        const int l = kl / sizes.nZeta;
                        const int k = kl % sizes.nZeta;
                        if (!is_close_rel_abs(
                                static_cast<double>(
                                    fourp.at("grpmn")[m][nf + n][k][l]),
                                scale * grpmn[idx_posn * sizes.nZnT + kl],
                                tol)) {
                            ok = false;
                        }
                        if (!is_close_rel_abs(
                                static_cast<double>(
                                    fourp.at("grpmn")[m][nf - n][k][l]),
                                scale * grpmn[idx_negn * sizes.nZnT + kl],
                                tol)) {
                            ok = false;
                        }
                    }
                }
            }
            check(ok, ("fourp grpmn (total) iter " + std::to_string(iter)));
        }

        ls.perform_toroidal_fourier_transforms();

        // vac1n_fouri: source, bcos/bsin, actemp/astemp (all 2*pi).
        {
            const json::Value fouri = load_golden(json_path("vac1n_fouri", iter));
            const double tol = 1e-9;
            const double scale = 2.0 * std::numbers::pi;

            const auto gstore_symm =
                to_host(ls.gstore_symm(), sizes.nThetaReduced * sizes.nZeta);
            bool ok = true;
            for (int l = 0; l < sizes.nThetaReduced; ++l) {
                for (int k = 0; k < sizes.nZeta; ++k) {
                    if (!is_close_rel_abs(
                            static_cast<double>(fouri.at("source")[k][l]),
                            scale * gstore_symm[l * sizes.nZeta + k], tol)) {
                        ok = false;
                    }
                }
            }
            check(ok, ("fouri source iter " + std::to_string(iter)));

            const auto bcos =
                to_host(ls.bcos(), (2 * nf + 1) * sizes.nThetaReduced);
            const auto bsin =
                to_host(ls.bsin(), (2 * nf + 1) * sizes.nThetaReduced);
            ok = true;
            for (int n = 0; n < nf + 1; ++n) {
                for (int l = 0; l < sizes.nThetaReduced; ++l) {
                    const int idx_posn = (nf + n) * sizes.nThetaReduced + l;
                    const int idx_negn = (nf - n) * sizes.nThetaReduced + l;
                    if (!is_close_rel_abs(
                            static_cast<double>(fouri.at("bcos")[l][nf + n]),
                            scale * bcos[idx_posn], tol)) {
                        ok = false;
                    }
                    if (!is_close_rel_abs(
                            static_cast<double>(fouri.at("bcos")[l][nf - n]),
                            scale * bcos[idx_negn], tol)) {
                        ok = false;
                    }
                    if (!is_close_rel_abs(
                            static_cast<double>(fouri.at("bsin")[l][nf + n]),
                            scale * bsin[idx_posn], tol)) {
                        ok = false;
                    }
                    if (!is_close_rel_abs(
                            static_cast<double>(fouri.at("bsin")[l][nf - n]),
                            scale * bsin[idx_negn], tol)) {
                        ok = false;
                    }
                }
            }
            check(ok, ("fouri bcos/bsin iter " + std::to_string(iter)));

            const auto actemp =
                to_host(ls.actemp(), mnpd * (2 * nf + 1) * sizes.nThetaEff);
            const auto astemp =
                to_host(ls.astemp(), mnpd * (2 * nf + 1) * sizes.nThetaEff);
            ok = true;
            for (int mn = 0; mn < mnpd; ++mn) {
                const int np = mn / (mf + 1);
                const int mp = mn % (mf + 1);
                for (int n = 0; n < nf + 1; ++n) {
                    for (int l = 0; l < sizes.nThetaEff; ++l) {
                        const int idx_posn =
                            (mn * (2 * nf + 1) + (nf + n)) * sizes.nThetaEff +
                            l;
                        if (!is_close_rel_abs(static_cast<double>(fouri.at(
                                                  "actemp")[mp][np][nf + n][l]),
                                              scale * actemp[idx_posn], tol)) {
                            ok = false;
                        }
                        if (!is_close_rel_abs(static_cast<double>(fouri.at(
                                                  "astemp")[mp][np][nf + n][l]),
                                              scale * astemp[idx_posn], tol)) {
                            ok = false;
                        }
                        if (n > 0) {
                            const int idx_negn =
                                (mn * (2 * nf + 1) + (nf - n)) *
                                    sizes.nThetaEff +
                                l;
                            if (!is_close_rel_abs(
                                    static_cast<double>(
                                        fouri.at("actemp")[mp][np][nf - n][l]),
                                    scale * actemp[idx_negn], tol)) {
                                ok = false;
                            }
                            if (!is_close_rel_abs(
                                    static_cast<double>(
                                        fouri.at("astemp")[mp][np][nf - n][l]),
                                    scale * astemp[idx_negn], tol)) {
                                ok = false;
                            }
                        }
                    }
                }
            }
            check(ok, ("fouri actemp/astemp iter " + std::to_string(iter)));
        }

        ls.perform_poloidal_fourier_transforms();
        ls.build_matrix();

        // vac1n_fouri bvec + amatrix and vac1n_solver potvac_in/amatrix
        // (scale 2*pi*(2*pi)^2); the golden bvec/amatrix compare the
        // singular-accumulated, gauge-zeroed quantities.
        {
            const json::Value fouri = load_golden(json_path("vac1n_fouri", iter));
            const json::Value solver =
                load_golden(json_path("vac1n_solver", iter));
            const double tol = 1e-9;
            const double scale = 2.0 * std::numbers::pi * 4.0 *
                                 std::numbers::pi * std::numbers::pi;

            // Assemble the RHS like SolveForPotential: bvec_sin +
            // singular/nfp, gauge-zeroed.
            const auto bvec_sin = to_host(ls.bvec_sin(), mnpd);
            const auto singular = to_host(si.bvec_sin(), mnpd);
            std::vector<double> bvec_full(mnpd);
            for (int mn = 0; mn < mnpd; ++mn) {
                bvec_full[mn] = bvec_sin[mn] + singular[mn] / sizes.nfp;
            }
            for (int all_n = 0; all_n < nf; ++all_n) {
                bvec_full[all_n * (mf + 1)] = 0.0;
            }

            bool ok = true;
            for (int mn = 0; mn < mnpd; ++mn) {
                const int np = mn / (mf + 1);
                const int mp = mn % (mf + 1);
                if (!is_close_rel_abs(
                        static_cast<double>(fouri.at("bvec")[mp][np]),
                        scale * bvec_full[mn], tol)) {
                    ok = false;
                }
                if (!is_close_rel_abs(
                        static_cast<double>(solver.at("potvac_in")[mn]),
                        scale * bvec_full[mn], tol)) {
                    ok = false;
                }
            }
            check(ok, ("bvec/potvac_in iter " + std::to_string(iter)));

            // amatrix: golden[row][col] vs matrix[col*mnpd + row] (the flat
            // layout vmecpp's LAPACK call consumed).
            const auto matrix = to_host(ls.matrix(), mnpd * mnpd);
            ok = true;
            for (int mn = 0; mn < mnpd; ++mn) {
                const int n = mn / (mf + 1);
                const int m = mn % (mf + 1);
                for (int mnp = 0; mnp < mnpd; ++mnp) {
                    const int np = mnp / (mf + 1);
                    const int mp = mnp % (mf + 1);
                    if (!is_close_rel_abs(
                            static_cast<double>(
                                fouri.at("amatrix")[m][n][mp][np]),
                            scale * matrix[mnp * mnpd + mn], tol)) {
                        ok = false;
                    }
                    if (!is_close_rel_abs(
                            static_cast<double>(solver.at("amatrix")[mn][mnp]),
                            scale * matrix[mnp * mnpd + mn], tol)) {
                        ok = false;
                    }
                }
            }
            check(ok, ("amatrix iter " + std::to_string(iter)));
        }

        // Solve and cross-check the potential against Fortran LAPACK.
        ls.decompose_matrix();
        ls.solve_for_potential(si.bvec_sin());
        {
            const json::Value solver =
                load_golden(json_path("vac1n_solver", iter));
            const double tol = 1e-9;
            const auto pot = to_host(ls.solution(), mnpd);
            bool ok = true;
            for (int mn = 0; mn < mnpd; ++mn) {
                if (!is_close_rel_abs(
                        static_cast<double>(solver.at("potvac_out")[mn]),
                        pot[mn], tol)) {
                    ok = false;
                }
            }
            check(ok, ("potvac_out (host LU vs Fortran LAPACK) iter " +
                       std::to_string(iter)));
        }
    }
}

}  // namespace

int main() {
    // The vmecpp flow carries the Laplace factor across iterations: build the
    // pipeline once, run the full update (53), then the partial update (54)
    // reusing the stale matrix/bvec_sin with the fresh singular RHS.
    {
        const json::Value vacuum = load_golden(json_path("vac1n_vacuum", 53));

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
        RegularizedIntegralsOperator<double> ri(sizes, sg);
        LaplaceSolverOperator<double> ls(sizes, fbd);

        const int nf = sizes.ntor;
        const int mf = sizes.mpol + 1;
        const int mnpd = (2 * nf + 1) * (mf + 1);

        const std::vector<double> raxis_53 =
            flat_array(vacuum.at("raxis_nestor"));
        const std::vector<double> zaxis_53 =
            flat_array(vacuum.at("zaxis_nestor"));
        const std::vector<double> extcur{4700.0, 1000.0};

        run_full_update(53, rcc, rss, zsc, zcs, raxis_53, zaxis_53, sg, ef, si,
                      ri, ls, extcur);

        // Partial update (iteration 54): fresh geometry/external/singular,
        // stale Laplace state, solve with the stale factor.
        const json::Value vacuum_54 = load_golden(json_path("vac1n_vacuum", 54));
        const int sign_j =
            static_cast<int>(static_cast<double>(vacuum_54.at("signgs")));
        std::vector<double> rcc_54(sizes.mnsize), rss_54(sizes.mnsize);
        fb.cos_to_cc_ss(flat_array(vacuum_54.at("rmnc")), rcc_54, rss_54,
                     sizes.ntor, sizes.mpol);
        std::vector<double> zsc_54(sizes.mnsize), zcs_54(sizes.mnsize);
        fb.sin_to_sc_cs(flat_array(vacuum_54.at("zmns")), zsc_54, zcs_54,
                     sizes.ntor, sizes.mpol);
        auto d_rcc = to_device(rcc_54);
        auto d_rss = to_device(rss_54);
        auto d_zsc = to_device(zsc_54);
        auto d_zcs = to_device(zcs_54);
        auto d_raxis = to_device(flat_array(vacuum_54.at("raxis_nestor")));
        auto d_zaxis = to_device(flat_array(vacuum_54.at("zaxis_nestor")));

        sg.update(d_rcc.data(), d_rss.data(), nullptr, nullptr, d_zsc.data(),
                  d_zcs.data(), nullptr, nullptr, sign_j, false);
        const double net_toroidal_current =
            static_cast<double>(vacuum_54.at("plascur")) /
            (4.0 * std::numbers::pi * 1.0e-7);
        ef.update(d_raxis.data(), d_zaxis.data(), net_toroidal_current);
        si.update(ef.b_dot_n(), false);
        ls.solve_for_potential(si.bvec_sin());

        const json::Value solver_54 = load_golden(json_path("vac1n_solver", 54));
        const double tol = 1e-9;
        const auto pot = to_host(ls.solution(), mnpd);
        bool ok = true;
        for (int mn = 0; mn < mnpd; ++mn) {
            if (!is_close_rel_abs(
                    static_cast<double>(solver_54.at("potvac_out")[mn]),
                    pot[mn], tol)) {
                ok = false;
            }
        }
        check(ok, "potvac_out iter 54 (stale factor + fresh RHS)");
    }

    return summary();
}
