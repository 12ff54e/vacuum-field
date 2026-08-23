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
using vfield::test::compareZetaFast;
using vfield::test::flatArray;
using vfield::test::is_close_rel_abs;
using vfield::test::loadGolden;
using vfield::test::nestedArray;
using vfield::test::summary;
using vfield::test::toDevice;
using vfield::test::toHost;

namespace {

const std::string kDataDir = "tests/data/cth_like_free_bdy/";

std::string jsonPath(const std::string& checkpoint, int iter) {
    std::ostringstream oss;
    oss << kDataDir << checkpoint << "/" << checkpoint << "_00015_"
        << std::setw(6) << std::setfill('0') << iter
        << "_01.cth_like_free_bdy.json";
    return oss.str();
}

void runIteration(int iter) {
    const json::Value vacuum = loadGolden(jsonPath("vac1n_vacuum", iter));
    const json::Value bextern = loadGolden(jsonPath("vac1n_bextern", iter));

    const bool full_update =
        (static_cast<double>(vacuum.at("ivac_skip")) == 0.0);
    const int sign_j =
        static_cast<int>(static_cast<double>(vacuum.at("signgs")));

    Sizes sizes(false, 5, 5, 4, 16, 36);
    FourierBasis fb(sizes);

    // Decode the LCFS inputs exactly like vmecpp's large tests.
    std::vector<double> rcc(sizes.mnsize), rss(sizes.mnsize);
    fb.cosToCcSs(flatArray(vacuum.at("rmnc")), rcc, rss, sizes.ntor,
                 sizes.mpol);
    std::vector<double> zsc(sizes.mnsize), zcs(sizes.mnsize);
    fb.sinToScCs(flatArray(vacuum.at("zmns")), zsc, zcs, sizes.ntor,
                 sizes.mpol);

    // cth_like_free_bdy: extcur = [4700, 1000] A (see the indata JSON).
    MgridProvider mgrid;
    mgrid.loadFile(kDataDir + "../mgrid_cth_like.nc",
                   std::vector<double>{4700.0, 1000.0});

    FourierBasisDevice<double> fbd(fb, sizes.lasym, sizes.nThetaEven);
    SurfaceGeometryOperator<double> sg(sizes, fbd);
    ExternalFieldOperator<double> ef(sizes, sg, mgrid);

    auto d_rcc = toDevice(rcc);
    auto d_rss = toDevice(rss);
    auto d_zsc = toDevice(zsc);
    auto d_zcs = toDevice(zcs);
    auto d_raxis = toDevice(flatArray(vacuum.at("raxis_nestor")));
    auto d_zaxis = toDevice(flatArray(vacuum.at("zaxis_nestor")));

    sg.update(d_rcc.data(), d_rss.data(), nullptr, nullptr, d_zsc.data(),
              d_zcs.data(), nullptr, nullptr, sign_j, full_update);

    // netToroidalCurrent = cTor / MU_0 (ideal_mhd_model.cc), plascur in A.
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
    const auto xpts = nestedArray(bextern.at("xpts_axis"));
    const auto axis_xyz =
        toHost(ef.axisXyz(), 3 * (sizes.nZeta * ef.nvper() + 1));
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
    check(compareZetaFast(nestedArray(bextern.at("mgrid_brad")),
                          toHost(ef.interpBr(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("mgrid_brad " + iter_name));
    check(compareZetaFast(nestedArray(bextern.at("mgrid_bphi")),
                          toHost(ef.interpBp(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("mgrid_bphi " + iter_name));
    check(compareZetaFast(nestedArray(bextern.at("mgrid_bz")),
                          toHost(ef.interpBz(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("mgrid_bz " + iter_name));

    // Axis-current contribution on its own (brad - mgrid_brad = curtorBr).
    const auto brad = nestedArray(bextern.at("brad"));
    const auto bphi = nestedArray(bextern.at("bphi"));
    const auto bz = nestedArray(bextern.at("bz"));
    const auto mgrid_brad = nestedArray(bextern.at("mgrid_brad"));
    const auto mgrid_bphi = nestedArray(bextern.at("mgrid_bphi"));
    const auto mgrid_bz = nestedArray(bextern.at("mgrid_bz"));
    const auto curtor_br = toHost(ef.curtorBr(), sizes.nZnT);
    const auto curtor_bp = toHost(ef.curtorBp(), sizes.nZnT);
    const auto curtor_bz = toHost(ef.curtorBz(), sizes.nZnT);
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
    check(compareZetaFast(nestedArray(bextern.at("bexu")),
                          toHost(ef.bSubU(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("bexu " + iter_name));
    check(compareZetaFast(nestedArray(bextern.at("bexv")),
                          toHost(ef.bSubV(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("bexv " + iter_name));
    check(compareZetaFast(nestedArray(bextern.at("bexn")),
                          toHost(ef.bDotN(), sizes.nZnT), sizes.nThetaEff,
                          tol) == 0,
          ("bexn " + iter_name));
}

}  // namespace

int main() {
    runIteration(53);
    runIteration(54);
    return summary();
}
