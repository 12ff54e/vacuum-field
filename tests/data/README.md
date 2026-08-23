# Test data provenance

All files in this directory are vendored from the vmecpp repository
(https://github.com/proximafusion/vmecpp, MIT license), tag v0.7.0, paths:

- `src/vmecpp/cpp/vmecpp_large_cpp_tests/test_data/cth_like_free_bdy/vac1n_*/`
  — Fortran-VMEC (educational_VMEC) NESTOR checkpoint dumps of a full
  free-boundary equilibrium run of the `cth_like_free_bdy` case (nfp=5,
  mpol=5, ntor=4, ntheta=16, nzeta=36, ns=15) at iterations 53 and 54.
  `vac1n_precal/fourp/fouri/greenf` exist only for iteration 53.
- `src/vmecpp/cpp/vmecpp_large_cpp_tests/test_data/boundary_coefficients_*csv`,
  `axis_coefficients_*.csv` — LCFS / axis Fourier coefficient tables of the
  same run (independent decode check for the golden inputs).
- `src/vmecpp/cpp/vmecpp/test_data/mgrid_cth_like.nc` — coil-field grid
  (101x101x36, 2 coil groups, field per ampere).
- `src/vmecpp/cpp/vmecpp/test_data/mgrid_solovev.nc` — axisymmetric coil
  field (201x201x1, 13 coil groups) for the `solovev_free_bdy` case.
- `src/vmecpp/cpp/vmecpp/test_data/cth_like_free_bdy.json`,
  `solovev_free_bdy.json` — the vmecpp INDATA inputs of both runs, for
  documentation of the case parameters.

The vmecpp files are distributed under the MIT license; see the repository
LICENSE. The golden values were produced by the Fortran VMEC reference
implementation and are used here only for numerical verification of this
library's port of the same algorithm.
