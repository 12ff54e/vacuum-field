// lu_solve.hpp — host dense LU solve (LAPACK dgetrf/dgetrs semantics).
//
// The dense Laplace system is small (mnpd = (2*ntor+1)(mpol+2), <= ~300 for
// realistic runs), so the factorization runs on the host in plain double —
// deterministic, dependency-free, and testable without a GPU. The routines
// replicate LAPACK's partial-pivoting factorization and triangular solves
// for COLUMN-major flat arrays; the Laplace solver passes its flat matrix
// in vmecpp's layout (which dgetrf/dgetrs consumed the same way), so the
// factorization behavior is identical to vmecpp's LAPACK call. If mnpd ever
// grows past a few hundred, this class is the seam where a cuSOLVER
// implementation would slot in.
#ifndef VFIELD_FREE_BOUNDARY_LU_SOLVE_HPP_
#define VFIELD_FREE_BOUNDARY_LU_SOLVE_HPP_

namespace vfield {

class LuSolve {
   public:
    // Partial-pivoting LU factorization in place (column-major a of size
    // n x n): on exit `a` holds L (unit diagonal, strict lower triangle) and
    // U (upper triangle); `pivots` holds the 1-based pivot row indices
    // (LAPACK convention). Returns the dgetrf-style info code: 0 on success,
    // k > 0 when the k-th pivot is exactly zero (singular matrix).
    static int decompose(double* a, int* pivots, int n);

    // Solves A x = b for the factorized A (dgetrs 'N' semantics): applies
    // the pivot permutation to b, forward-substitutes L y = P b, then
    // back-substitutes U x = y. x is returned in b.
    static void solve(const double* a, const int* pivots, double* b, int n);
};

}  // namespace vfield

#endif  // VFIELD_FREE_BOUNDARY_LU_SOLVE_HPP_
