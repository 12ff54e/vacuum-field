// test_singular_integrals.cu — T_l^+/- recurrences vs Gauss-Legendre.
//
// Port of vmecpp's TlpTlmAccuracyTest: the operator's prepareUpdate runs on
// synthetic CONSTANT geometry (a=0.8, b2=-1.3, c=1.2, so am/ap ~ 4.7) and the
// resulting T tables are compared per point against 64-point Gauss-Legendre
// quadrature of the defining integral. Two (mpol, ntor) resolutions cover
// the forward-stable regime (kL small) and the Miller backward branch
// (kL large enough that forward would lose > 10 digits).
#include "vfield/common/fourier_basis.hpp"
#include "vfield/common/fourier_basis_device.cuh"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/singular_integrals_operator.hpp"
#include "vfield/free_boundary/surface_geometry_operator.hpp"
#include "vfield_test_cuda_helper.cuh"

#include <array>
#include <cmath>
#include <vector>

using vfield::FourierBasis;
using vfield::FourierBasisDevice;
using vfield::SingularIntegralsOperator;
using vfield::Sizes;
using vfield::SurfaceGeometryOperator;
using vfield::test::check;
using vfield::test::is_close_rel_abs;
using vfield::test::summary;
using vfield::test::toDevice;
using vfield::test::toHost;

namespace {

// 64-point Gauss-Legendre quadrature on [-1, 1], {weight, abscissa} pairs.
constexpr int kGLNodes = 64;
constexpr std::array<std::array<double, 2>, kGLNodes> kGaussLegendre64 = {{
    {0.0486909570091397, -0.0243502926634244},
    {0.0486909570091397, 0.0243502926634244},
    {0.0485754674415034, -0.0729931217877990},
    {0.0485754674415034, 0.0729931217877990},
    {0.0483447622348030, -0.1214628192961206},
    {0.0483447622348030, 0.1214628192961206},
    {0.0479993885964583, -0.1696444204239928},
    {0.0479993885964583, 0.1696444204239928},
    {0.0475401657148303, -0.2174236437400071},
    {0.0475401657148303, 0.2174236437400071},
    {0.0469681828162100, -0.2646871622087674},
    {0.0469681828162100, 0.2646871622087674},
    {0.0462847965813144, -0.3113228719902110},
    {0.0462847965813144, 0.3113228719902110},
    {0.0454916279274181, -0.3572201583376681},
    {0.0454916279274181, 0.3572201583376681},
    {0.0445905581637566, -0.4022701579639916},
    {0.0445905581637566, 0.4022701579639916},
    {0.0435837245293235, -0.4463660172534641},
    {0.0435837245293235, 0.4463660172534641},
    {0.0424735151236536, -0.4894031457070530},
    {0.0424735151236536, 0.4894031457070530},
    {0.0412625632426235, -0.5312794640198946},
    {0.0412625632426235, 0.5312794640198946},
    {0.0399537411327203, -0.5718956462026340},
    {0.0399537411327203, 0.5718956462026340},
    {0.0385501531786156, -0.6111553551723933},
    {0.0385501531786156, 0.6111553551723933},
    {0.0370551285402400, -0.6489654712546573},
    {0.0370551285402400, 0.6489654712546573},
    {0.0354722132568824, -0.6852363130542333},
    {0.0354722132568824, 0.6852363130542333},
    {0.0338051618371416, -0.7198818501716109},
    {0.0338051618371416, 0.7198818501716109},
    {0.0320579283548516, -0.7528199072605319},
    {0.0320579283548516, 0.7528199072605319},
    {0.0302346570724025, -0.7839723589433414},
    {0.0302346570724025, 0.7839723589433414},
    {0.0283396726142595, -0.8132653151227975},
    {0.0283396726142595, 0.8132653151227975},
    {0.0263774697150547, -0.8406292962525803},
    {0.0263774697150547, 0.8406292962525803},
    {0.0243527025687109, -0.8659993981540928},
    {0.0243527025687109, 0.8659993981540928},
    {0.0222701738083833, -0.8893154459951141},
    {0.0222701738083833, 0.8893154459951141},
    {0.0201348231535302, -0.9105221370785028},
    {0.0201348231535302, 0.9105221370785028},
    {0.0179517157756973, -0.9295691721319396},
    {0.0179517157756973, 0.9295691721319396},
    {0.0157260304760247, -0.9464113748584028},
    {0.0157260304760247, 0.9464113748584028},
    {0.0134630478967186, -0.9610087996520538},
    {0.0134630478967186, 0.9610087996520538},
    {0.0111681394601311, -0.9733268277899110},
    {0.0111681394601311, 0.9733268277899110},
    {0.0088467598263639, -0.9833362538846260},
    {0.0088467598263639, 0.9833362538846260},
    {0.0065044579689784, -0.9910133714767443},
    {0.0065044579689784, 0.9910133714767443},
    {0.0041470332605625, -0.9963401167719553},
    {0.0041470332605625, 0.9963401167719553},
    {0.0017832807216964, -0.9993050417357722},
    {0.0017832807216964, 0.9993050417357722},
}};

// Reference T^{+/-}_l via numerical quadrature of the defining integral
//   T^+_l = integral_{-1}^{+1} t^l / sqrt(am + 2*d*t + ap*t^2) dt
//   T^-_l = integral_{-1}^{+1} t^l / sqrt(ap + 2*d*t + am*t^2) dt
// (the integrands are smooth on [-1,+1] as long as the discriminant
// d^2 - ap*am < 0, i.e. b2 != 0).
double tlpReference(int l, double ap, double am, double d) {
    double sum = 0.0;
    for (const auto& [w, t] : kGaussLegendre64) {
        const double tl = std::pow(t, l);
        const double radicand = am + 2.0 * d * t + ap * t * t;
        sum += w * tl / std::sqrt(radicand);
    }
    return sum;
}

double tlmReference(int l, double ap, double am, double d) {
    // T^-_l swaps ap <-> am in the radicand.
    double sum = 0.0;
    for (const auto& [w, t] : kGaussLegendre64) {
        const double tl = std::pow(t, l);
        const double radicand = ap + 2.0 * d * t + am * t * t;
        sum += w * tl / std::sqrt(radicand);
    }
    return sum;
}

template <class T>
void runResolution(int mpol,
                   int ntor,
                   double a_val,
                   double b2_val,
                   double c_val,
                   double tol) {
    const bool lasym = false;
    const int nfp = 5;
    const int ntheta = 0;
    // Sizes requires nzeta >= 2*ntor+4 to resolve all toroidal modes; use a
    // small multiple.
    const int nzeta = 4 * (ntor + 1);

    Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);
    FourierBasis fb(sizes);
    FourierBasisDevice<T> fbd(fb, sizes.lasym, sizes.nThetaEven);
    SurfaceGeometryOperator<T> sg(sizes, fbd);
    SingularIntegralsOperator<T> si(sizes, fbd, sg);

    const int nf = ntor;
    const int mf = mpol + 1;
    const int kL = mf + nf;

    // The vmecpp coefficients (0.8, -1.3, 1.2) give am/ap ~ 4.7 and the
    // discriminant d^2 - ap*am < 0 (smooth integrand); the forward
    // recurrence loses ~kL * log10(am/ap) digits and the Miller branch
    // fires when > 10 digits would be lost. Float runs use near-degenerate
    // coefficients instead (the vmecpp geometry would lose ~6 of float's 7
    // digits already at kL=9).
    const double ap = a_val + b2_val + c_val;
    const double am = a_val - b2_val + c_val;
    const double d = c_val - a_val;

    std::vector<T> a(sizes.nZnT, static_cast<T>(a_val));
    std::vector<T> b2(sizes.nZnT, static_cast<T>(b2_val));
    std::vector<T> c(sizes.nZnT, static_cast<T>(c_val));
    std::vector<T> zero(sizes.nZnT, static_cast<T>(0));

    auto d_a = toDevice(a);
    auto d_b2 = toDevice(b2);
    auto d_c = toDevice(c);
    auto d_zero = toDevice(zero);

    // full_update=false: the second-fundamental-form inputs are unused.
    si.prepareUpdate(d_a.data(), d_b2.data(), d_c.data(), d_zero.data(),
                     d_zero.data(), d_zero.data(), false);

    const auto tlp = toHost(si.tlp(), (kL + 2) * sizes.nZnT);
    const auto tlm = toHost(si.tlm(), (kL + 2) * sizes.nZnT);

    bool ok = true;
    for (int l = 0; l <= kL; ++l) {
        const double Tp_ref = tlpReference(l, ap, am, d);
        const double Tm_ref = tlmReference(l, ap, am, d);
        for (int kl = 0; kl < sizes.nZnT; ++kl) {
            if (!is_close_rel_abs(Tp_ref,
                                  static_cast<double>(tlp[l * sizes.nZnT + kl]),
                                  tol)) {
                ok = false;
            }
            if (!is_close_rel_abs(Tm_ref,
                                  static_cast<double>(tlm[l * sizes.nZnT + kl]),
                                  tol)) {
                ok = false;
            }
        }
    }
    const std::string label =
        std::string("T recurrences vs Gauss-Legendre (mpol=") +
        std::to_string(mpol) + ", ntor=" + std::to_string(ntor) +
        ", kL=" + std::to_string(kL) + ")";
    check(ok, label);
}

}  // namespace

int main() {
    // kL=9: forward recurrence is accurate (forward branch).
    runResolution<double>(4, 4, 0.8, -1.3, 1.2, 1e-11);
    // kL=17: the Miller backward branch fires (forward would lose > 10
    // digits).
    runResolution<double>(8, 8, 0.8, -1.3, 1.2, 1e-11);
    // Float: near-degenerate coefficients (the vmecpp geometry loses ~6 of
    // float's 7 digits already at kL=9) and a relaxed tolerance.
    runResolution<float>(4, 4, 0.8, 0.1, 1.2, 1e-5);
    runResolution<float>(8, 8, 0.8, 0.1, 1.2, 1e-5);
    return summary();
}
