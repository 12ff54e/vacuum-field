// golden_data.hpp — helpers for reading the vendored Fortran-VMEC dumps.
//
// The vac1n_*.json files in tests/data are nested float arrays in Fortran
// memory order: toroidal index k first, poloidal index l second ([k][l]).
// The library's real-space arrays are zeta-fast (kl = l*nZeta + k); the
// comparison helpers below take the JSON value and a flat host array and
// remap accordingly.
#ifndef VFIELD_TESTS_SUPPORT_GOLDEN_DATA_HPP_
#define VFIELD_TESTS_SUPPORT_GOLDEN_DATA_HPP_

#include "vfield/JsonParser.h"
#include "vfield_test_cuda_helper.cuh"

#include <cstddef>
#include <string>
#include <vector>

namespace vfield::test {

// Loads and parses a golden JSON file.
inline json::Value loadGolden(const std::string& path) {
    return json::parse_file(path);
}

// Flat double array json[i].
inline std::vector<double> flatArray(const json::Value& v) {
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        out[i] = static_cast<double>(v[i]);
    }
    return out;
}

// Two-level double array json[i][j] (the Fortran dumps' [k][l] nesting).
inline std::vector<std::vector<double>> nestedArray(const json::Value& v) {
    std::vector<std::vector<double>> out;
    out.reserve(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        out.push_back(flatArray(v[i]));
    }
    return out;
}

// Compares a nested [k][l] golden array against a zeta-fast flat host array
// over the given number of poloidal rows, with the row length inferred from
// the golden's inner size. Returns the number of failing points.
template <class T>
std::size_t compareZetaFast(const std::vector<std::vector<double>>& golden,
                            const std::vector<T>& actual,
                            std::size_t n_rows,
                            double tol) {
    const std::size_t n_zeta = golden.size();
    const std::size_t n_l = golden[0].size();
    std::size_t failures = 0;
    for (std::size_t l = 0; l < n_rows && l < n_l; ++l) {
        for (std::size_t k = 0; k < n_zeta; ++k) {
            const double got = static_cast<double>(actual[l * n_zeta + k]);
            if (!is_close_rel_abs(got, golden[k][l], tol)) { ++failures; }
        }
    }
    return failures;
}

}  // namespace vfield::test

#endif  // VFIELD_TESTS_SUPPORT_GOLDEN_DATA_HPP_
