# vacuum-field

CUDA-accelerated free-boundary vacuum magnetic field solver for VMEC-style
equilibria. A port of vmecpp's NESTOR algorithm (MIT-licensed, Proxima Fusion,
https://github.com/proximafusion/vmecpp at tag v0.7.0) into a standalone
CUDA C++ library with no dependency on either vmecpp or cuMES.

## What it computes

Given the last-closed-flux-surface (LCFS) Fourier coefficients and an external
coil field (an `mgrid` file), the solver computes the vacuum magnetic field on
the plasma boundary consistent with the coil field:

- magnetic scalar potential Φ on the LCFS, solved from the boundary-element
  (Green's second identity) form of the vacuum Laplace problem,
  `Φ/2 + PV∫ Φ ∂G/∂n′ = ∫ G ∂Φ/∂n′`, discretized in the sin(mθ−nζ) Fourier
  basis (dense system, dimension `(2·ntor+1)(mpol+2)`);
- covariant field components `b_sub_u`/`b_sub_v` (potential + external field);
- vacuum magnetic pressure `b_sq_vac = |B|²/2` and cylindrical components
  `B_R`/`B_φ`/`B_Z` on the boundary;
- the surface-integral scalars `b_sub_u_vac`/`b_sub_v_vac` (net toroidal / poloidal
  current diagnostics).

The singular kernel is split into an analytically integrable local
approximation (closed-form one-dimensional recurrences) plus a regularized
smooth remainder evaluated by direct quadrature over the surface and the `nfp`
field-period images.

## Build

```bash
cmake --preset verify-double     # C++20 + CUDA, -Werror, Release
cmake --build build -j
ctest --test-dir build --output-on-failure

cmake --preset float             # float instantiation (experimental)
cmake --build build-float -j
```

Requirements: CUDA Toolkit >= 11, CMake >= 3.20. Optional: NetCDF C library
(`libnetcdf-dev`) for `mgrid` file reading (auto-detected; disable with
`-DVFIELD_USE_NETCDF=OFF`).

## Verification

Numerical correctness is verified elementwise against Fortran-VMEC reference
data for a full free-boundary equilibrium run (vmecpp's `cth_like_free_bdy`
case, `tests/data/`): surface geometry at 1e-12, external field at 1e-10,
singular/regularized integrals and the Laplace pipeline at 1e-9/5e-10, and the
end-to-end `b_sq_vac` at 1e-10. See `docs/verification.md`.

## License

MIT. The algorithm is ported from vmecpp (Copyright (c) 2024-present Proxima
Fusion GmbH), MIT-licensed; see LICENSE.
