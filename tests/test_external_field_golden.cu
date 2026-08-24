// test_external_field_golden.cu — vac1n_bextern comparison (double).
//
// Golden gate: decodes the LCFS + axis inputs from the vac1n_vacuum dumps,
// loads the mgrid coil field, runs the operator, and compares the mgrid
// interpolation, the axis-current contribution, the full field, and the
// covariant/normal components elementwise against the Fortran-VMEC
// vac1n_bextern dumps at 1e-10 (the tolerance vmecpp's own large test uses
// for this module). Both iterations (53 and 54) are compared.
#include "golden_data.hpp"
#include "vfield/common/fourier_basis.hpp"
#include "vfield/common/fourier_basis_device.cuh"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/external_field_operator.hpp"
#include "vfield/free_boundary/mgrid_provider.hpp"
#include "vfield/free_boundary/surface_geometry_operator.hpp"

#include <iomanip>
#include <numbers>
#include <sstream>
#include <string>

using vfield::ExternalFieldOperator;
using vfield::FourierBasis;
using vfield::FourierBasisDevice;
using vfield::MgridProvider;
using vfield::Sizes;
using vfield::SurfaceGeometryOperator;
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

void run_iteration(int iter) {
    const json::Value vacuum = load_golden(json_path("vac1n_vacuum", iter));
    const json::Value bextern = load_golden(json_path("vac1n_bextern", iter));

    const bool full_update =
        (static_cast<double>(vacuum.at("ivac_skip")) == 0.0);
    const int sign_j =
        static_cast<int>(static_cast<double>(vacuum.at("signgs")));

    Sizes sizes(false, 5, 5, 4, 16, 36);
    FourierBasis fb(sizes);

    // Decode the LCFS inputs exactly like vmecpp's large tests.
    std::vector<double> rcc(sizes.mnsize), rss(sizes.mnsize);
    fb.cos_to_cc_ss(flat_array(vacuum.at("rmnc")), rcc, rss, sizes.ntor,
                 sizes.mpol);
    std::vector<double> zsc(sizes.mnsize), zcs(sizes.mnsize);
    fb.sin_to_sc_cs(flat_array(vacuum.at("zmns")), zsc, zcs, sizes.ntor,
                 sizes.mpol);

    // cth_like_free_bdy: extcur = [4700, 1000] A (see the indata JSON).
    MgridProvider mgrid;
    mgrid.load_file(DATA_DIR + "../mgrid_cth_like.nc",
                   std::vector<double>{4700.0, 1000.0});

    FourierBasisDevice<double> fbd(fb, sizes.lasym, sizes.nThetaEven);
    SurfaceGeometryOperator<double> sg(sizes, fbd);
    ExternalFieldOperator<double> ef(sizes, sg, mgrid);

    auto d_rcc = to_device(rcc);
    auto d_rss = to_device(rss);
    auto d_zsc = to_device(zsc);
    auto d_zcs = to_device(zcs);
    auto d_raxis = to_device(flat_array(vacuum.at("raxis_nestor")));
    auto d_zaxis = to_device(flat_array(vacuum.at("zaxis_nestor")));

    sg.update(d_rcc.data(), d_rss.data(), nullptr, nullptr, d_zsc.data(),
              d_zcs.data(), nullptr, nullptr, sign_j, full_update);

    // net_toroidal_current = cTor / MU_0 (ideal_mhd_model.cc), plascur in A.
    const double net_toroidal_current =
        static_cast<double>(vacuum.at("plascur")) /
        (4.0 * std::numbers::pi * 1.0e-7);
    ef.update(d_raxis.data(), d_zaxis.data(), net_toroidal_current);

    const double tol = 1e-10;
    const std::string iter_name = "iter " + std::to_string(iter);

    check(is_close_rel_abs(static_cast<double>(bextern.at("axis_current")),
                           net_toroidal_current, tol),
          ("axis_current " + iter_name));

    // Axis polygon, first field-period module [3][nZeta] and the closure.
    const auto xpts = nested_array(bextern.at("xpts_axis"));
    const auto axis_xyz =
        to_host(ef.axis_xyz(), 3 * (sizes.nZeta * ef.nvper() + 1));
    bool axis_ok = true;
    for (int k = 0; k < sizes.nZeta; ++k) {
        for (int j = 0; j < 3; ++j) {
            if (!is_close_rel_abs(xpts[j][k], axis_xyz[3 * k + j], tol)) {
                axis_ok = false;
            }
        }
    }
    check(axis_ok, ("xpts_axis " + iter_name));

    // mgrid interpolation on its own.
    check(compare_zeta_fast(nested_array(bextern.at("mgrid_brad")),
                          to_host(ef.interp_br(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("mgrid_brad " + iter_name));
    check(compare_zeta_fast(nested_array(bextern.at("mgrid_bphi")),
                          to_host(ef.interp_bp(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("mgrid_bphi " + iter_name));
    check(compare_zeta_fast(nested_array(bextern.at("mgrid_bz")),
                          to_host(ef.interp_bz(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("mgrid_bz " + iter_name));

    // Axis-current contribution on its own (brad - mgrid_brad = curtor_br).
    const auto brad = nested_array(bextern.at("brad"));
    const auto bphi = nested_array(bextern.at("bphi"));
    const auto bz = nested_array(bextern.at("bz"));
    const auto mgrid_brad = nested_array(bextern.at("mgrid_brad"));
    const auto mgrid_bphi = nested_array(bextern.at("mgrid_bphi"));
    const auto mgrid_bz = nested_array(bextern.at("mgrid_bz"));
    const auto curtor_br = to_host(ef.curtor_br(), sizes.nZnT);
    const auto curtor_bp = to_host(ef.curtor_bp(), sizes.nZnT);
    const auto curtor_bz = to_host(ef.curtor_bz(), sizes.nZnT);
    bool curtor_ok = true;
    for (int l = 0; l < sizes.nThetaEff; ++l) {
        for (int k = 0; k < sizes.nZeta; ++k) {
            const int kl = l * sizes.nZeta + k;
            curtor_ok &= is_close_rel_abs(brad[k][l] - mgrid_brad[k][l],
                                          curtor_br[kl], tol);
            curtor_ok &= is_close_rel_abs(bphi[k][l] - mgrid_bphi[k][l],
                                          curtor_bp[kl], tol);
            curtor_ok &=
                is_close_rel_abs(bz[k][l] - mgrid_bz[k][l], curtor_bz[kl], tol);
        }
    }
    check(curtor_ok, ("axis-current contribution " + iter_name));

    // Quantities derived from the full field.
    check(compare_zeta_fast(nested_array(bextern.at("bexu")),
                          to_host(ef.b_sub_u(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("bexu " + iter_name));
    check(compare_zeta_fast(nested_array(bextern.at("bexv")),
                          to_host(ef.b_sub_v(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("bexv " + iter_name));
    check(compare_zeta_fast(nested_array(bextern.at("bexn")),
                          to_host(ef.b_dot_n(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("bexn " + iter_name));
}

}  // namespace

int main() {
    run_iteration(53);
    run_iteration(54);
    return summary();
}
