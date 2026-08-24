// fourier_basis.hpp — host-side Fourier basis tables for the vacuum solver.
//
// A port of vmecpp's FourierBasis restricted to the NESTOR (zeta-fast,
// n-major) layout:
//   poloidal basis index (m, l) = l * (mnyq2 + 1) + m
//   toroidal basis index (n, k) = n * nZeta + k
//   product coefficient index (m, n) = n * m_size + m   (n-major)
// The basis values carry the DFT normalization split into mscale/nscale
// (1 for m == 0 / n == 0, sqrt(2) otherwise): one sqrt(2) enters the forward
// (geometry-into-realspace) transform and one the inverse (forces-into-Fourier)
// transform. cosmui/sinmui are the poloidal integration weights
// (intNorm = 1/(nZeta * (nThetaReduced - 1)), endpoints halved) times the
// basis values; the toroidal derivatives are taken w.r.t. the geometric
// toroidal angle phi = zeta / nfp (cosnvn = n * nfp * cosnv).
#ifndef VFIELD_COMMON_FOURIER_BASIS_HPP_
#define VFIELD_COMMON_FOURIER_BASIS_HPP_

#include "vfield/common/sizes.hpp"

#include <span>
#include <vector>

namespace vfield {

// -1 if x<0, 0 if x==0, +1 if x>0
inline int signum(int x) {
    return (x > 0) - (x < 0);
}

class FourierBasis {
   public:
    explicit FourierBasis(const Sizes& sizes);

    // Index helpers mirroring the zeta-fast layout (static so both host code
    // and device code can lay out index arithmetic identically).
    static int poloidal_basis_index(int m, int l, int num_m) {
        return l * num_m + m;
    }
    static int toroidal_basis_index(int n, int k, int num_k) {
        return n * num_k + k;
    }
    // (m, n) -> n * m_size + m in the n-major coefficient layout.
    static int product_index(int m, int n, int m_size) { return n * m_size + m; }

    // Convert cos(xm[mn] theta - xn[mn] zeta) spectral arrays into the 2D
    // (n-major) product-basis arrays of the NESTOR coefficient layout.
    // Returns the number of converted modes (mnmax).
    int cos_to_cc_ss(std::span<const double> fc_cos,
                  std::span<double> fc_cc,
                  std::span<double> fc_ss,
                  int n_size,
                  int m_size) const;
    int sin_to_sc_cs(std::span<const double> fc_sin,
                  std::span<double> fc_sc,
                  std::span<double> fc_cs,
                  int n_size,
                  int m_size) const;

    // Round-trips (used by the unit tests and the golden-input decode check).
    int cc_ss_to_cos(std::span<const double> fc_cc,
                  std::span<const double> fc_ss,
                  std::span<double> fc_cos,
                  int n_size,
                  int m_size) const;
    int sc_cs_to_sin(std::span<const double> fc_sc,
                  std::span<const double> fc_cs,
                  std::span<double> fc_sin,
                  int n_size,
                  int m_size) const;

    std::vector<double> mscale;  // [mnyq2 + 1]
    std::vector<double> nscale;  // [nnyq2 + 1]

    // [nThetaReduced * (mnyq2 + 1)]
    std::vector<double> cosmu;
    std::vector<double> sinmu;
    std::vector<double> cosmum;  // m * cosmu
    std::vector<double> sinmum;  // -m * sinmu
    std::vector<double> cosmui;  // integration-weighted
    std::vector<double> sinmui;  // integration-weighted

    // [(nnyq2 + 1) * nZeta]
    std::vector<double> cosnv;
    std::vector<double> sinnv;
    std::vector<double> cosnvn;  // n * nfp * cosnv (d/dphi)
    std::vector<double> sinnvn;  // -n * nfp * sinnv (d/dphi)

    const Sizes& sizes() const { return sizes_; }

   private:
    void compute_fourier_basis(int nfp);

    Sizes sizes_;
};

}  // namespace vfield

#endif  // VFIELD_COMMON_FOURIER_BASIS_HPP_
