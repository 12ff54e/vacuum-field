// singular_coefficients.cpp — cmn/cmns expansion coefficients.
//
// Direct port of vmecpp's SingularIntegrals::computeCoefficients (Algorithm 1
// and eq. 6.291 of "the numerics of VMEC++"). The recurrence and the layout
// must stay verbatim: the golden tests compare against them.
#include "vfield/common/singular_coefficients.hpp"

#include <algorithm>
#include <cstdlib>

namespace vfield {

SingularCoefficients::SingularCoefficients(int nf, int mf) : nf_(nf), mf_(mf) {
    cmn.assign((1 + nf + mf) * (nf + 1) * (mf + 1), 0.0);
    cmns.assign((1 + nf + mf) * (nf + 1) * (mf + 1), 0.0);
    computeCoefficients();
}

void SingularCoefficients::computeCoefficients() {
    // below loop sets only parts of cmn,
    // so initialize all entries to zero once here
    std::fill(cmn.begin(), cmn.end(), 0.0);

    // cmn from scratch: Algorithm 1 in TNOV
    for (int n = 0; n < nf_ + 1; ++n) {
        for (int m = 0; m < mf_ + 1; ++m) {
            int i_mn = m - n;
            int j_mn = m + n;
            int k_mn = abs(i_mn);

            // originally: s_mn = (j_mn + k_mn) / 2
            // (j+k) is always even, so dividing by 2 is always possible
            // also: s_mn = 0.5*(m + n + abs(m - n)) == max(m, n)
            int s_mn = std::max(m, n);

            double f1 = 1.0;
            double f2 = 1.0;
            double f3 = 1.0;

            for (int i = 1; i <= k_mn; ++i) {
                f1 *= s_mn - i + 1;
                f2 *= i;
            }

            // (-1)^{(l-i_mn)/2} == (-1)^{(k_mn-i_mn)/2} at beginning of
            // l-loop; note (6.182 in TNOV) that (k_mn - i_mn) / 2 ==
            // max(0, n - m)
            // --> compute initial value and then reverse on each iteration of
            // l loop
            // l gets increased by 2 --> l/2 gets increased by 1 per iteration
            // --> (-1)^{(l-i_mn)/2} == (-1)^{l/2 - i_mn/2} == (-1)^{l/2} /
            // (-1)^{i_mn/2} and since i_mn is constant during the
            // l-iterations, the sign reversed in each iteration
            int cmnSign = (std::max(0, n - m) % 2 == 0) ? 1 : -1;

            for (int l = k_mn; l <= j_mn; l += 2) {
                int lnm = (l * (nf_ + 1) + n) * (mf_ + 1) + m;

                cmn[lnm] = f1 / (f2 * f3) * cmnSign;

                f1 *= (l + 2 + j_mn) * (j_mn - l) * 0.25;
                f2 *= (l + 2 + k_mn) * 0.5;
                f3 *= (l + 2 - k_mn) * 0.5;

                cmnSign = -cmnSign;
            }  // l
        }  // m
    }  // n

    // cmns from cmn: (6.291) in TNOV
    for (int n = 0; n < nf_ + 1; ++n) {
        for (int m = 0; m < mf_ + 1; ++m) {
            int n_m_ = n * (mf_ + 1) + m;
            int n1m_ = (n - 1) * (mf_ + 1) + m;
            int n_m1 = n * (mf_ + 1) + (m - 1);
            int n1m1 = (n - 1) * (mf_ + 1) + (m - 1);
            for (int l = 0; l < 1 + mf_ + nf_; ++l) {
                int ln_m_ = l * (mf_ + 1) * (nf_ + 1) + n_m_;
                int ln1m_ = l * (mf_ + 1) * (nf_ + 1) + n1m_;
                int ln_m1 = l * (mf_ + 1) * (nf_ + 1) + n_m1;
                int ln1m1 = l * (mf_ + 1) * (nf_ + 1) + n1m1;
                if (m == 0 && n == 0) {
                    cmns[ln_m_] = cmn[ln_m_];
                } else if (m == 0 && n > 0) {
                    cmns[ln_m_] = (cmn[ln_m_] + cmn[ln1m_]) / 2;
                } else if (m > 0 && n == 0) {
                    cmns[ln_m_] = (cmn[ln_m_] + cmn[ln_m1]) / 2;
                } else {
                    cmns[ln_m_] =
                        (cmn[ln_m_] + cmn[ln1m_] + cmn[ln_m1] + cmn[ln1m1]) / 2;
                }
            }  // l
        }  // m
    }  // n
}

}  // namespace vfield
