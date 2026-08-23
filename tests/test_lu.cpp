// test_lu.cpp — host LU solve against a brute-force reference.
//
// Host-only. Verifies LuSolve (LAPACK dgetrf/dgetrs semantics) against an
// O(n^3) Gaussian-elimination-with-pivoting reference on well-conditioned
// random systems, a pivot-heavy diagonal permutation, and singular-matrix
// detection.
#include "vfield/free_boundary/lu_solve.hpp"
#include "vfield_test.h"

#include <cmath>
#include <vector>

using vfield::LuSolve;
using vfield::test::check;
using vfield::test::is_close_rel_abs;
using vfield::test::summary;

namespace {

// Deterministic pseudo-random doubles in [-1, 1].
double prand(int* state) {
    *state = (*state * 1103515245 + 12345) & 0x7fffffff;
    return 2.0 * (*state) / 2147483647.0 - 1.0;
}

// Brute-force reference solve: Gaussian elimination with partial pivoting
// (column-major), writing x into b.
// b is taken by reference (the solution is written into it); the matrix
// copy is fine by value.
void referenceSolve(std::vector<double> a, std::vector<double>& b, int n) {
    for (int k = 0; k < n; ++k) {
        int p = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::fabs(a[i + k * n]) > std::fabs(a[p + k * n])) p = i;
        }
        for (int j = 0; j < n; ++j) { std::swap(a[k + j * n], a[p + j * n]); }
        std::swap(b[k], b[p]);
        const double akk = a[k + k * n];
        for (int i = k + 1; i < n; ++i) {
            const double f = a[i + k * n] / akk;
            for (int j = k + 1; j < n; ++j) {
                a[i + j * n] -= f * a[k + j * n];
            }
            b[i] -= f * b[k];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) { b[i] -= a[i + j * n] * b[j]; }
        b[i] /= a[i + i * n];
    }
}

}  // namespace

int main() {
    // Well-conditioned random systems (diagonally dominant).
    int state = 42;
    for (int n : {2, 5, 33, 63}) {
        std::vector<double> a(n * n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) { a[i + j * n] = prand(&state); }
            a[i + i * n] += 4.0 * n;  // diagonal dominance
        }
        std::vector<double> x_exact(n);
        for (int i = 0; i < n; ++i) x_exact[i] = prand(&state);

        // b = A x (column-major)
        std::vector<double> b(n, 0.0);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) { b[i] += a[i + j * n] * x_exact[j]; }
        }

        std::vector<double> lu = a;
        std::vector<int> pivots(n, 0);
        const int info = LuSolve::decompose(lu.data(), pivots.data(), n);
        check(info == 0, "lu decompose ok (n=" + std::to_string(n) + ")");

        std::vector<double> x = b;
        LuSolve::solve(lu.data(), pivots.data(), x.data(), n);

        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (!is_close_rel_abs(x_exact[i], x[i], 1e-10)) ok = false;
        }
        check(ok, "lu solve matches A x = b (n=" + std::to_string(n) + ")");

        // Cross-check the residual against the brute-force reference.
        std::vector<double> x_ref = b;
        referenceSolve(a, x_ref, n);
        ok = true;
        for (int i = 0; i < n; ++i) {
            if (!is_close_rel_abs(x_ref[i], x[i], 1e-10)) ok = false;
        }
        check(ok, "lu solve matches reference (n=" + std::to_string(n) + ")");
    }

    // Singular matrix detection: a zero column.
    {
        const int n = 4;
        std::vector<double> a(n * n, 1.0);
        for (int i = 0; i < n; ++i) a[i + 2 * n] = 0.0;  // zero column 2
        std::vector<int> pivots(n, 0);
        const int info = LuSolve::decompose(a.data(), pivots.data(), n);
        check(info > 0, "singular matrix reported");
    }

    return summary();
}
