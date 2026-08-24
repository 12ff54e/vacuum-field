# vacuum-field — algorithm

A port of vmecpp's NESTOR free-boundary vacuum solver (MIT, Proxima Fusion,
tag v0.7.0). See vmecpp's `docs/the_numerics_of_vmecpp.pdf` ("TNOV") for the
derivations; the equations below follow the numbering there.

## Problem

Given the LCFS Fourier coefficients and the external coil field (mgrid),
find the vacuum magnetic field on the plasma boundary consistent with the
coils: solve the Neumann Laplace problem for the magnetic scalar potential
with `B_vac = grad(Phi)` in the region between the boundary S and the
coils, `dPhi/dn = -B_ext . n` on S. The boundary-element (Green's second
identity) form is an integral equation of the second kind,

    Phi/2 + PV int_S Phi(x') dG/dn' dS' = int_S G dPhi/dn' dS'

with the free-space Green's function `G = 1/(4 pi |x - x'|)` summed over
the nfp field-period images. Discretized in the `sin(m*theta - n*zeta)`
basis, this is the dense system `A . Phi = b` of dimension
`mnpd = (2*ntor+1)(mpol+2)`; the Phi/2 jump term is the `+0.5` diagonal.

## Singular / regularized split

The `1/|x - x'|` kernel is split at each point into an analytic local
approximation in tan-half-angle variables (built from the metric's local
quadratic form) plus a smooth remainder:

- `SingularIntegralsOperator`: the closed-form part — the one-dimensional
  integrals `T_l^+/-` (three-term recurrence, eq. 6.207; Miller's backward
  recurrence when the forward branch would lose > 10 digits) and the
  surface-weighted `S_l^+/-` (eq. A17) with the cmn/cmns expansion
  coefficients (Algorithm 1 / eq. 6.291).
- `RegularizedIntegralsOperator`: the difference kernel evaluated by direct
  quadrature over the surface and the nfp field-period images (`greenp`,
  the normal-derivative kernel; `gstore`, the source). The exact
  singularity is skipped (it lives in the singular part). For
  axisymmetric plasmas (nZeta == 1) the toroidal integral runs over
  nvper = 64 toroidal images.

## Assembly and solve

`LaplaceSolverOperator` analyses the regularized kernels into the
sin(m*theta - n*zeta) basis (odd part under the (theta, zeta) -> (-theta,
-zeta) reflection), adds the singular part scaled by 1/nfp, performs the
toroidal and poloidal DFTs (direct matrix-vector form — the mode counts
are tiny), and assembles the dense matrix: the duplicated m=0/n<0 gauge
rows are zeroed and the RHS entries for those modes set to zero. The dense
system is factorized once per full update and solved every update; both
run on the host in double (LAPACK dgetrf/dgetrs semantics on the same flat
layout vmecpp passes to LAPACK).

## Outputs

From the potential coefficients: `pot_u/pot_v = dPhi/dtheta, dPhi/dphi` on
the boundary (basis de-normalized by mscale/nscale), the covariant vacuum
field `b_sub_u/b_sub_v = pot + external covariant`, the surface integrals
`b_sub_u_vac/b_sub_v_vac = signJ * 2 pi * int b_sub_u/V wInt dtheta`, and — through
the full-torus metric (guv*nfp/2, gvv*nfp^2) — the contravariant
components, `b_sq_vac = |B|^2/2` (no mu0, the VMEC force convention) and the
cylindrical components `B_R/B_phi/B_Z`.

## CUDA mapping

Every operator is a device-buffer-owning class with one thread per output
element, replaying vmecpp's accumulation order exactly (no atomics, no
cuFFT — the analysis DFTs are smaller than FFT launch overhead). Only the
dense solve runs on the host; the seam for a future cuSOLVER backend is
`vfield::LuSolve`.
