# CLAUDE.md — vacuum-field

## Project overview

`vacuum-field` is a standalone CUDA C++ library: a port of vmecpp's NESTOR
free-boundary vacuum-field algorithm (MIT; vmecpp at tag v0.7.0) with no
dependency on either vmecpp or cuMES. It computes the vacuum magnetic field
on the LCFS from the boundary Fourier coefficients and an mgrid coil field:
the magnetic scalar potential via the boundary-element Laplace solve, and
the outputs `b_sq_vac` (|B|²/2, no mu0), the covariant components
`b_sub_u`/`b_sub_v`, the cylindrical `B_R/B_phi/B_Z`, and the
surface-integral scalars `b_sub_u_vac`/`b_sub_v_vac`. Algorithm details in
`docs/algorithm.md`; the test/golden matrix in `docs/verification.md`.

## Build & run

```bash
cmake --preset verify-double   # precise math, -Werror
cmake --build build -j
ctest --test-dir build --output-on-failure

cmake --preset float           # float instantiation (smoke only)
cmake --build build-float -j
```

**Requirements:** CUDA Toolkit >= 11, CMake >= 3.20; NetCDF (optional, for
mgrid reading). C++20 strict throughout. When embedded via
`add_subdirectory` (cuMES), the CUDA host compiler and architectures are
defaulted only-if-unset — the consumer sets them before `project()`.

## Coding conventions

### Naming

- **Types:** `PascalCase` (e.g., `SurfaceGeometryOperator`,
  `VacuumFieldSolver`)
- **Functions:** `snake_case` (e.g., `solve_for_potential`,
  `upload_table_extended`); `__global__` kernels keep a `_kernel` suffix
- **Variables:** `snake_case` (e.g., `d_rcc`, `n_rows`, `full_update`);
  compact physics/Fortran-derived abbreviations are exempt and stay as-is
  (`nZnT`, `nZeta`, `nThetaEven`, `wInt`, `signJ`, `cTor`, `bigNo`,
  `epsTan`, `intNorm`, `nSq`/`mSq`, `klRel`/`klRev`/`lRev`-style index
  names)
- **Constants:** `CAPITAL_SNAKE_CASE` (e.g., `BLOCK_SIZE`, `SIGN_J`,
  `TWO_PI`, `LOG_GROWTH_THRESHOLD`) — no `kCamelCase` remains
- **Templated types:** every templated struct/class aliases its scalar type
  parameter as `using val_type = T;` (first public member)
- **Host pointers:** raw pointers in host code only where absolutely
  necessary (CUDA/C-library interop and device-side kernel/operator
  members). Everything else: `std::vector`/`std::span`/`std::string_view`,
  `std::optional<std::reference_wrapper<T>>` for nullable params,
  `DeviceBuffer` for device allocations (including test-harness staging)
- **Device pointers:** `d_` prefix; **host pointers:** `h_` prefix

### Layout

- `include/vfield/` — the public contract (host-side types and the
  operator-class headers); kernel bodies live in `src/kernels/<mod>_impl.cuh`,
  included only by the `src/<mod>_{double,float}.cu` explicit-instantiation
  TUs (one scalar type per TU, so kernels may declare dynamic shared memory
  directly as `extern __shared__ T[]`)
- Host-side modules are plain C++ under `src/vfield/`
- All device allocations via RAII (`DeviceBuffer`), error-checked through
  the centralized `vfield::check_cuda` in `vfield/runtime`
- Include guards are `#ifndef` guards, not `#pragma once`
- Namespace `vfield`; tests in namespace `vfield::test`

### Tests

- Standalone executables, no framework; CPU reference mirrors GPU kernel
  logic (pattern: known input → GPU → CPU reference → compare)
- Golden tests compare elementwise against the vendored Fortran-VMEC dumps
  in `tests/data/` at vmecpp's own tolerances (see `docs/verification.md`)
- The dense solve runs on the host in double (`LuSolve`, LAPACK dgetrf/
  dgetrs semantics) — the documented seam for a future cuSOLVER backend

## Known deviations

- GPU FMA contraction vs non-FMA CPU references: per-element gaps
  <= ~1e-12, far below every golden tolerance (calibrated into
  `test_regularized_axisym`'s 1e-11).
- Float: recurrence thresholds tuned for double; float is covered by
  smoke/parity tests only (the dense solve stays double in float builds).
- The golden data covers only the stellarator-symmetric path; lasym is
  gated by the parity test.
