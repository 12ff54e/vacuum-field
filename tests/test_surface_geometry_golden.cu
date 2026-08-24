// test_surface_geometry_golden.cu — vac1n_surface comparison (double).
//
// Golden gate: decodes the LCFS inputs from the vac1n_vacuum dumps (rmnc ->
// r_cc/r_ss, zmns -> z_sc/z_cs via the FourierBasis conversions — the same decode
// vmecpp's large tests use), runs the operator, and compares every array
// elementwise against the Fortran-VMEC vac1n_surface dumps at 1e-12. Both
// checkpoints (iterations 53 and 54) are compared; iteration 54 has
// ivac_skip = 1, so the second derivatives and full-update arrays are only
// compared for iteration 53.
#include "golden_data.hpp"
#include "vfield/common/fourier_basis.hpp"
#include "vfield/common/fourier_basis_device.cuh"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/surface_geometry_operator.hpp"

#include <iomanip>
#include <sstream>
#include <string>

using vfield::FourierBasis;
using vfield::FourierBasisDevice;
using vfield::Sizes;
using vfield::SurfaceGeometryOperator;
using vfield::test::check;
using vfield::test::compare_zeta_fast;
using vfield::test::flat_array;
using vfield::test::load_golden;
using vfield::test::nested_array;
using vfield::test::summary;
using vfield::test::to_device;
using vfield::test::to_host;

namespace {

const std::string DATA_DIR = "tests/data/cth_like_free_bdy/";

std::string vacuum_path(int iter) {
    std::ostringstream oss;
    oss << DATA_DIR << "vac1n_vacuum/vac1n_vacuum_00015_" << std::setw(6)
        << std::setfill('0') << iter << "_01.cth_like_free_bdy.json";
    return oss.str();
}

std::string surface_path(int iter) {
    std::ostringstream oss;
    oss << DATA_DIR << "vac1n_surface/vac1n_surface_00015_" << std::setw(6)
        << std::setfill('0') << iter << "_01.cth_like_free_bdy.json";
    return oss.str();
}

void run_iteration(int iter) {
    const json::Value vacuum = load_golden(vacuum_path(iter));
    const json::Value surf = load_golden(surface_path(iter));

    const bool full_update =
        (static_cast<double>(vacuum.at("ivac_skip")) == 0.0);
    const int sign_j =
        static_cast<int>(static_cast<double>(vacuum.at("signgs")));

    // cth_like_free_bdy resolution (see tests/data/README.md).
    Sizes sizes(false, 5, 5, 4, 16, 36);
    FourierBasis fb(sizes);

    // Decode the LCFS inputs exactly like vmecpp's large tests.
    std::vector<double> rcc(sizes.mnsize), rss(sizes.mnsize);
    fb.cos_to_cc_ss(flat_array(vacuum.at("rmnc")), rcc, rss, sizes.ntor,
                 sizes.mpol);
    std::vector<double> zsc(sizes.mnsize), zcs(sizes.mnsize);
    fb.sin_to_sc_cs(flat_array(vacuum.at("zmns")), zsc, zcs, sizes.ntor,
                 sizes.mpol);

    FourierBasisDevice<double> fbd(fb, sizes.lasym, sizes.nThetaEven);
    SurfaceGeometryOperator<double> sg(sizes, fbd);
    auto d_rcc = to_device(rcc);
    auto d_rss = to_device(rss);
    auto d_zsc = to_device(zsc);
    auto d_zcs = to_device(zcs);

    sg.update(d_rcc.data(), d_rss.data(), nullptr, nullptr, d_zsc.data(),
              d_zcs.data(), nullptr, nullptr, sign_j, full_update);

    const double tol = 1e-12;
    const std::string iter_name = "iter " + std::to_string(iter);

    // r1b/z1b over the effective poloidal range only (vmecpp's own large test
    // compares [0, nZnT); the mirrored second half is only refreshed on full
    // updates and is covered by the unit test's direct-series comparison).
    check(compare_zeta_fast(nested_array(surf.at("r1b")),
                          to_host(sg.r1b(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("r1b " + iter_name));
    check(compare_zeta_fast(nested_array(surf.at("z1b")),
                          to_host(sg.z1b(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("z1b " + iter_name));

    // Reduced-range arrays [k][l] over the nThetaEff rows.
    check(compare_zeta_fast(nested_array(surf.at("rub")),
                          to_host(sg.rub(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("rub " + iter_name));
    check(compare_zeta_fast(nested_array(surf.at("rvb")),
                          to_host(sg.rvb(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("rvb " + iter_name));
    check(compare_zeta_fast(nested_array(surf.at("zub")),
                          to_host(sg.zub(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("zub " + iter_name));
    check(compare_zeta_fast(nested_array(surf.at("zvb")),
                          to_host(sg.zvb(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("zvb " + iter_name));
    check(compare_zeta_fast(nested_array(surf.at("snr")),
                          to_host(sg.snr(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("snr " + iter_name));
    check(compare_zeta_fast(nested_array(surf.at("snv")),
                          to_host(sg.snv(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("snv " + iter_name));
    check(compare_zeta_fast(nested_array(surf.at("snz")),
                          to_host(sg.snz(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("snz " + iter_name));
    check(compare_zeta_fast(nested_array(surf.at("guu_b")),
                          to_host(sg.guu(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("guu " + iter_name));
    check(compare_zeta_fast(nested_array(surf.at("guv_b")),
                          to_host(sg.guv(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("guv " + iter_name));
    check(compare_zeta_fast(nested_array(surf.at("gvv_b")),
                          to_host(sg.gvv(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("gvv " + iter_name));

    // Full-update arrays: only compared when the checkpoint is a full update.
    if (full_update) {
        check(compare_zeta_fast(nested_array(surf.at("ruu")),
                              to_host(sg.ruu(), sizes.nZnT), sizes.nThetaEff,
                              tol) == 0,
              ("ruu " + iter_name));
        check(compare_zeta_fast(nested_array(surf.at("ruv")),
                              to_host(sg.ruv(), sizes.nZnT), sizes.nThetaEff,
                              tol) == 0,
              ("ruv " + iter_name));
        check(compare_zeta_fast(nested_array(surf.at("rvv")),
                              to_host(sg.rvv(), sizes.nZnT), sizes.nThetaEff,
                              tol) == 0,
              ("rvv " + iter_name));
        check(compare_zeta_fast(nested_array(surf.at("zuu")),
                              to_host(sg.zuu(), sizes.nZnT), sizes.nThetaEff,
                              tol) == 0,
              ("zuu " + iter_name));
        check(compare_zeta_fast(nested_array(surf.at("zuv")),
                              to_host(sg.zuv(), sizes.nZnT), sizes.nThetaEff,
                              tol) == 0,
              ("zuv " + iter_name));
        check(compare_zeta_fast(nested_array(surf.at("zvv")),
                              to_host(sg.zvv(), sizes.nZnT), sizes.nThetaEff,
                              tol) == 0,
              ("zvv " + iter_name));
        check(compare_zeta_fast(nested_array(surf.at("auu")),
                              to_host(sg.auu(), sizes.nZnT), sizes.nThetaEff,
                              tol) == 0,
              ("auu " + iter_name));
        check(compare_zeta_fast(nested_array(surf.at("auv")),
                              to_host(sg.auv(), sizes.nZnT), sizes.nThetaEff,
                              tol) == 0,
              ("auv " + iter_name));
        check(compare_zeta_fast(nested_array(surf.at("avv")),
                              to_host(sg.avv(), sizes.nZnT), sizes.nThetaEff,
                              tol) == 0,
              ("avv " + iter_name));
        check(compare_zeta_fast(nested_array(surf.at("drv")),
                              to_host(sg.drv(), sizes.nZnT), sizes.nThetaEff,
                              tol) == 0,
              ("drv " + iter_name));
        check(compare_zeta_fast(nested_array(surf.at("rzb2")),
                              to_host(sg.rzb2(), sizes.nThetaEven * sizes.nZeta),
                              sizes.nThetaEven, tol) == 0,
              ("rzb2 " + iter_name));
        check(
            compare_zeta_fast(nested_array(surf.at("rcosuv")),
                            to_host(sg.rcosuv(), sizes.nThetaEven * sizes.nZeta),
                            sizes.nThetaEven, tol) == 0,
            ("rcosuv " + iter_name));
        check(
            compare_zeta_fast(nested_array(surf.at("rsinuv")),
                            to_host(sg.rsinuv(), sizes.nThetaEven * sizes.nZeta),
                            sizes.nThetaEven, tol) == 0,
            ("rsinuv " + iter_name));
    }
}

}  // namespace

int main() {
    run_iteration(53);
    run_iteration(54);
    return summary();
}
