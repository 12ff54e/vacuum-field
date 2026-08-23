# vacuum-field — verification

The library is verified elementwise against Fortran-VMEC (educational_VMEC)
checkpoint dumps of a full free-boundary equilibrium run of the
`cth_like_free_bdy` case (nfp=5, mpol=5, ntor=4, ntheta=16, nzeta=36,
ns=15), vendored in `tests/data/` with provenance. The golden inputs are
decoded from the `vac1n_vacuum` dumps (rmnc/zmns -> rCC/rSS/zSC/zCS via the
FourierBasis conversions — the same decode vmecpp's own large tests use),
which makes each comparison self-contained per iteration.

| Test | Compares | Tolerance |
| ---- | -------- | --------- |
| `test_sizes`, `test_fourier_basis` | derived counts, weights, basis scales/orthogonality, cos/sin round trips (host, CI-runnable) | exact / 1e-14 |
| `test_lu` | host LU vs a brute-force reference, singular detection | 1e-10 |
| `test_surface_geometry` | operator vs a direct trigonometric CPU reference, double+float | 1e-12 / 1e-4 |
| `test_surface_geometry_golden` | `vac1n_surface` (r1b/z1b, derivatives, snr/snv/snz, guu/guv/gvv, ruu..zvv, auu/auv/avv, drv, rzb2, rcosuv/rsinuv), iterations 53/54 | 1e-12 |
| `test_external_field_golden` | `vac1n_bextern` (axis polygon, mgrid interpolation, axis-current contribution, bexu/bexv/bexn), iterations 53/54 | 1e-10 |
| `test_singular_coefficients` | cmn vs the factorial formula; cmns vs `vac1n_precal` (alp factor, [l][m][n] layout) | 1e-12 / 1e-14 |
| `test_singular_integrals` | T_l^+/- recurrences vs 64-point Gauss-Legendre, forward and Miller branches, double+float | 1e-11 / 1e-5 |
| `test_singular_golden` | `vac1n_analyt` (all_tlp/tlm, all_slp/slm, bvec at alp*(2 pi)^2, grpmn at alp), iterations 53/54 | 1e-9 |
| `test_regularized_golden` | `vac1n_greenf` (tanu/tanv, gstore at 4 pi^2/alp, greenp at 1/alp), iteration 53 | 5e-10 |
| `test_regularized_axisym` | axisymmetric nvper=64 path vs a CPU mirror of updateAxisymmetric | 1e-11 / 1e-4 |
| `test_laplace_solver` | manufactured single-mode round trip of the analysis DFTs (incl. lasym) | 1e-12 / 1e-4 |
| `test_laplace_golden` | `vac1n_fourp/fouri/solver` (grpmn, source, bcos/bsin, actemp/astemp, bvec, amatrix, potvac_in; potvac_out cross-validates the host LU against Fortran LAPACK), iterations 53/54 | 1e-9 |
| `test_vacuum_field_golden` | `vac1n_bsqvac` (potsin, potu/potv, bsubu/bsubv, bsqvac, brv/bphiv/bzv) — full end-to-end NESTOR equivalence, iterations 53/54 | 1e-10 |
| `test_float_smoke` | float pipeline vs the library's own double run | 1e-4 / 1e-3 |
| `test_lasym_parity` | lasym with zero antisymmetric coefficients == symmetric result (axis-current driven; no Fortran golden exists for lasym) | 1e-12 |

Scale factors: the Fortran dumps carry `alp = 2*pi/nfp` in cmns/grpmn (and
`(2 pi)^2` in the bvec source terms), which the library's internal
normalizations do not — the golden tests apply exactly the factors vmecpp's
own large tests document.

## Known deviations

- **GPU FMA contraction**: the kernels contract multiply-adds while the CPU
  references do not; per-element gaps are <= ~1e-12 (measured 1.35e-12 on
  the axisymmetric greenp comparison), far below every golden tolerance.
  The `test_regularized_axisym` double tolerance (1e-11) is calibrated for
  this.
- **Float**: the Miller activation thresholds and the golden tolerances are
  tuned for double; float is covered by smoke/parity tests at relaxed
  tolerances (the dense solve stays double in float builds).
- **lasym**: the golden data covers only the stellarator-symmetric path;
  lasym is gated by the parity test and vmecpp's own lasym round-trip
  fixture (vmecpp's lasym support is likewise incomplete upstream).
