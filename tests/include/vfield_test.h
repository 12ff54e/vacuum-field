// vfield_test.h — CUDA-free test harness (assertions, comparison, summary).
//
// Included by every test: directly by the host-only .cpp tests, transitively
// by the .cu tests via vfield_test_cuda_helper.cuh. CUDA-free (no
// cuda_runtime); ostream-only so no std::format toolchain dependence.
#ifndef VFIELD_TESTS_INCLUDE_VFIELD_TEST_H_
#define VFIELD_TESTS_INCLUDE_VFIELD_TEST_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>

namespace vfield::test {

// The single shared failure counter (function-local static, so header-only and
// shared across TUs). Capture `int before = failures();` to count only the
// failures a sub-check adds.
inline int& failures() {
    static int n = 0;
    return n;
}

// Assert `ok`; on failure print "FAIL <msg>" and increment the counter.
inline void check(bool ok, std::string_view msg) {
    if (ok) {
        std::cout << "PASS " << msg << '\n';
    } else {
        std::cout << "FAIL " << msg << '\n';
        ++failures();
    }
}

// Max element-wise |a[i] - b[i]| over the shorter of the two ranges. Accepts
// any contiguous range (std::vector, std::array, std::span, raw C arrays) via
// the generic RangeA/RangeB deduction.
template <typename RangeA, typename RangeB>
double max_diff(const RangeA& a, const RangeB& b) {
    const std::size_t n = std::min(a.size(), b.size());
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

// Pointer + length convenience (raw device->host buffers).
template <typename T>
double max_diff(const T* a, const T* b, std::size_t n) {
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

// Assert |a - b| <= tol; on failure print the values and the message.
inline void expect_near(double a, double b, double tol, std::string_view msg) {
    if (std::fabs(a - b) <= tol) {
        std::cout << "PASS " << msg << '\n';
    } else {
        std::cout << "FAIL " << msg << " (a=" << a << " b=" << b
                  << " diff=" << std::fabs(a - b) << ")\n";
        ++failures();
    }
}

// |a - b| <= tol * (1 + |a| + |b|): vmecpp's IsCloseRelAbs.
inline bool is_close_rel_abs(double a, double b, double tol) {
    return std::fabs(a - b) <= tol * (1.0 + std::fabs(a) + std::fabs(b));
}

// Max element-wise relative-absolute difference over the shorter of the two
// ranges (the range form of is_close_rel_abs).
template <typename RangeA, typename RangeB>
double max_rel_diff(const RangeA& a, const RangeB& b) {
    const std::size_t n = std::min(a.size(), b.size());
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double diff =
            std::fabs(double(a[i]) - double(b[i])) /
            (1.0 + std::fabs(double(a[i])) + std::fabs(double(b[i])));
        m = std::max(m, diff);
    }
    return m;
}

// Print the final summary and return the process exit code (0 = all pass).
inline int summary() {
    if (failures()) {
        std::cout << failures() << " FAILURES\n";
        return 1;
    }
    std::cout << "ALL PASS\n";
    return 0;
}

}  // namespace vfield::test

#endif  // VFIELD_TESTS_INCLUDE_VFIELD_TEST_H_
