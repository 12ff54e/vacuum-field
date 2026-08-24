// test_regularized_golden.cu — vac1n_greenf comparison (double).
//
// Golden gate: runs the full surface-geometry + external-field +
// regularized-integrals pipeline on the cth-like inputs and compares the
// tan-half-angle tables (vac1n_precal, direct), the greenp kernel (scale
// 1/(2 pi/nfp), Fortran [k][l][kp][lp] layout), and gstore (scale
// 4 pi^2/(2 pi/nfp), Fortran [k][l] layout) against the Fortran-VMEC dumps
// at 5e-10 (vmecpp's own tolerance for this module). The greenf dump exists
// only for the full-update iteration 53.
#include "golden_data.hpp"
#include "vfield/common/fourier_basis.hpp"
#include "vfield/common/fourier_basis_device.cuh"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/external_field_operator.hpp"
#include "vfield/free_boundary/mgrid_provider.hpp"
#include "vfield/free_boundary/regularized_integrals_operator.hpp"
#include "vfield/free_boundary/surface_geometry_operator.hpp"

#include <iomanip>
#include <numbers>
#include <sstream>
#include <string>

using vfield::ExternalFieldOperator;
using vfield::FourierBasis;
using vfield::FourierBasisDevice;
using vfield::MgridProvider;
using vfield::RegularizedIntegralsOperator;
using vfield::Sizes;
using vfield::SurfaceGeometryOperator;
using vfield::test::check;
using vfield::test::flat_array;
using vfield::test::is_close_rel_abs;
using vfield::test::load_golden;
using vfield::test::summary;
using vfield::test::to_device;
using vfield::test::to_host;

const std::string DATA_DIR = "tests/data/cth_like_free_bdy/";

int main() {
    const int iter = 53;
    const json::Value vacuum = load_golden(
        DATA_DIR +
        "vac1n_vacuum/vac1n_vacuum_00015_000053_01.cth_like_free_bdy.json");
    const json::Value greenf = load_golden(
        DATA_DIR +
        "vac1n_greenf/vac1n_greenf_00015_000053_01.cth_like_free_bdy.json");
    const json::Value precal = load_golden(
        DATA_DIR +
        "vac1n_precal/vac1n_precal_00015_000053_01.cth_like_free_bdy.json");

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
    RegularizedIntegralsOperator<double> ri(sizes, sg);

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
    ri.update(ef.b_dot_n());

    const double tol = 5e-10;

    // tanu/tanv tables (vac1n_precal, direct comparison).
    const auto tanu = to_host(ri.tanu(), sizes.nThetaEven);
    const auto tanv = to_host(ri.tanv(), sizes.nZeta);
    bool tan_ok = true;
    for (int l = 0; l < sizes.nThetaEven; ++l) {
        if (!is_close_rel_abs(static_cast<double>(precal.at("tanu_1d")[l]),
                              tanu[l], 1e-14)) {
            tan_ok = false;
        }
    }
    for (int k = 0; k < sizes.nZeta; ++k) {
        if (!is_close_rel_abs(static_cast<double>(precal.at("tanv_1d")[k]),
                              tanv[k], 1e-14)) {
            tan_ok = false;
        }
    }
    check(tan_ok, "tanu/tanv vs vac1n_precal");

    // gstore: scale 4 pi^2 / (2 pi/nfp), Fortran [k][l] layout.
    const double gstore_scale = 4.0 * std::numbers::pi * std::numbers::pi /
                                (2.0 * std::numbers::pi / sizes.nfp);
    const auto gstore = to_host(ri.gstore(), sizes.nThetaEven * sizes.nZeta);
    bool gstore_ok = true;
    for (int kl = 0; kl < sizes.nThetaEven * sizes.nZeta; ++kl) {
        const int l = kl / sizes.nZeta;
        const int k = kl % sizes.nZeta;
        if (!is_close_rel_abs(static_cast<double>(greenf.at("gstore")[k][l]),
                              gstore_scale * static_cast<double>(gstore[kl]),
                              tol)) {
            gstore_ok = false;
        }
    }
    check(gstore_ok, "gstore");

    // greenp: scale 1/(2 pi/nfp), Fortran [k][l][kp][lp] layout.
    const double greenp_scale = 1.0 / (2.0 * std::numbers::pi / sizes.nfp);
    const auto greenp_host =
        to_host(ri.greenp(), sizes.nZnT * sizes.nThetaEven * sizes.nZeta);
    bool greenp_ok = true;
    for (int klp = 0; klp < sizes.nZnT; ++klp) {
        const int lp = klp / sizes.nZeta;
        const int kp = klp % sizes.nZeta;
        for (int kl = 0; kl < sizes.nThetaEven * sizes.nZeta; ++kl) {
            const int l = kl / sizes.nZeta;
            const int k = kl % sizes.nZeta;
            const int ip = klp * sizes.nThetaEven * sizes.nZeta + kl;
            if (!is_close_rel_abs(
                    static_cast<double>(greenf.at("greenp")[k][l][kp][lp]),
                    greenp_scale * static_cast<double>(greenp_host[ip]), tol)) {
                greenp_ok = false;
            }
        }
    }
    check(greenp_ok, "greenp");

    (void)iter;
    return summary();
}
