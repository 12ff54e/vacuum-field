// lu_solve.cpp — host dense LU solve (LAPACK dgetrf/dgetrs semantics).
//
// The pivot search keeps the FIRST maximal |entry| (LAPACK uses a strict >
// comparison), the row swap covers the whole row, and the permutation is
// applied to the right-hand side by sequential swaps in pivot order —
// exactly what dlaswp does. This makes the factorization behavior
// equivalent to vmecpp's LAPACK call on the same flat array.
#include "vfield/free_boundary/lu_solve.hpp"

#include <cmath>
#include <stdexcept>

namespace vfield {

int LuSolve::decompose(double* a, int* pivots, int n) {
    for (int k = 0; k < n; ++k) {
        // find the pivot row: first occurrence of the maximal magnitude
        int p = k;
        double max_abs = std::fabs(a[k + k * n]);
        for (int i = k + 1; i < n; ++i) {
            const double v = std::fabs(a[i + k * n]);
            if (v > max_abs) {
                max_abs = v;
                p = i;
            }
        }

        if (a[p + k * n] == 0.0) {
            // singular matrix: U(k,k) is exactly zero
            return k + 1;
        }

        pivots[k] = p + 1;

        // swap rows p and k over all columns
        if (p != k) {
            for (int j = 0; j < n; ++j) {
                const double tmp = a[k + j * n];
                a[k + j * n] = a[p + j * n];
                a[p + j * n] = tmp;
            }
        }

        // scale the subdiagonal of column k
        if (k + 1 < n) {
            const double akk = a[k + k * n];
            for (int i = k + 1; i < n; ++i) { a[i + k * n] /= akk; }
        }

        // update the trailing submatrix
        for (int j = k + 1; j < n; ++j) {
            const double akj = a[k + j * n];
            for (int i = k + 1; i < n; ++i) {
                a[i + j * n] -= a[i + k * n] * akj;
            }
        }
    }
    return 0;
}

void LuSolve::solve(const double* a, const int* pivots, double* b, int n) {
    // apply the pivot permutation (dlaswp semantics: sequential swaps)
    for (int i = 0; i < n; ++i) {
        const int ip = pivots[i] - 1;
        if (ip != i) {
            const double tmp = b[i];
            b[i] = b[ip];
            b[ip] = tmp;
        }
    }

    // forward substitution: L y = P b (unit diagonal)
    for (int j = 0; j < n; ++j) {
        if (b[j] != 0.0) {
            for (int i = j + 1; i < n; ++i) { b[i] -= a[i + j * n] * b[j]; }
        }
    }

    // backward substitution: U x = y
    for (int j = n - 1; j >= 0; --j) {
        b[j] /= a[j + j * n];
        for (int i = 0; i < j; ++i) { b[i] -= a[i + j * n] * b[j]; }
    }
}

}  // namespace vfield
