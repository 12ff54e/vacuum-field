// singular_coefficients.hpp — cmn/cmns expansion coefficients (host).
//
// Port of vmecpp's SingularIntegrals::compute_coefficients: the expansion
// coefficients of the singular-kernel angular dependence (cmn, TNOV
// Algorithm 1) and their surface-weighted average (cmns, TNOV eq. 6.291).
// Layout: ((l * (nf + 1) + n) * (mf + 1) + m) with l in [0, nf+mf].
#ifndef VFIELD_COMMON_SINGULAR_COEFFICIENTS_HPP_
#define VFIELD_COMMON_SINGULAR_COEFFICIENTS_HPP_

#include <cstddef>
#include <vector>

namespace vfield {

class SingularCoefficients {
   public:
    SingularCoefficients(int nf, int mf);

    // ((l * (nf + 1) + n) * (mf + 1) + m), l in [0, nf + mf].
    std::vector<double> cmn;
    std::vector<double> cmns;

   private:
    void compute_coefficients();

    int nf_;
    int mf_;
};

}  // namespace vfield

#endif  // VFIELD_COMMON_SINGULAR_COEFFICIENTS_HPP_
