# Emission intercomparison Version 2: Stage 1

## Outcome

Version-2 Stage 1 establishes the frozen WASP-17b common contract and reruns
the grey/isothermal closure with scattering exactly disabled.  The typed,
immutable contract is serialized in
`docs/data/emission_intercomparison/version_2/common_contract.json`; reusable
pressure, stellar, R=100, isothermal, and PG14 arrays are in
`version_2_common_profiles.npz`.

The active contract was revised to `0.3--12 micron` with 369 R=100 bins and
PICASO correlated-k only. Stages 1--3 are regenerated because their serialized
arrays, normalization products, metrics, and integrity records depend on the
spectral domain. The numerical gates below remain frozen and unchanged before
the revised matrices are inspected. Historical `0.8--12 micron` results remain
recoverable in Git history and are not reinterpreted.

The full matrix completed safely and preserved all results, but its overall
scientific status is an **out-of-tolerance closure regime** because one of nine predeclared gates is not met.
The failed threshold was not relaxed after inspection.  Eight gates pass,
including the genuinely compatible ROBERT/pRT shared-tensor comparison,
vertical convergence, exact-zero handling, and the separately declared
angular relative limit.

## Provenance and common contract

The annotated `emission-intercomparison-v1` tag peels to the documented
Version-1 endpoint `f00e0616c7aae7d37e0badda295c189ead17dde1`, which is an
ancestor of the Version-2 base `53d438648c34c08d7e89696b182cd61aee3384b9`.
The machine-readable `version_lineage.json` records that relationship, the
separate manuscript-repository exclusion, and that the large raw-output tree
was neither copied, moved, nor recursively hashed.  The separate manuscript
repository was not entered or modified.

The common contract contains and tests:

- the adopted TEPCat/NASA WASP-17 measurements, uncertainties, frozen SI
  conversions, constants, gravity, radius/area ratios, and irradiation checks;
- exact `6550 K` Planck surface flux in `W m-2 m-1`, evaluated with `expm1`;
- 40, 80, and 160 pressure cells from `1e-5` to `100 bar`, including edges,
  geometric centres, orientation, PICASO levels, and pRT nodes;
- the `1755 K` isothermal array and both frozen PG14 families at every
  resolution, with implementation and method SHA-256 values;
- the exact six VMRs, H2/He remainder rule, provenance state, molecular masses,
  and computed mean molecular weight;
- exact 0.3--12 micron comparison endpoints, 369 equal-log R=100 bins,
  flux-conserving integration, and native-array retention rules; and
- the PICASO correlated-k-only rule and retired opacity-sampling state.

PICASO correlated-k is mandatory from Stage 2 onward.  The common contract
records the four official PICASO-4 resort-rebin HDF5 assets from Zenodo DOI
`10.5281/zenodo.18644980`, Version 2, with the supplied SHA-256 values.  All
share 661 spectral bins, eight double-Gauss points, 20-by-73 PT arrays, and
coverage of `0.267868--267.559 micron`, `1e-6--3000 bar`, and `75--4000 K`.
The parent task reported exact verification against both SHA-256 and official
MD5 records; Stage 1 did not recursively rehash Dropbox storage.

The dedicated Version-2 environment uses PICASO 4.0 on Python 3.11.15 and the
official Version-4 reference tree from Git commit
`0369089372f748609dd0233e6de9361af31a38cf`.  Its complete frozen package,
reference-file checksum, worker-environment, and passed correlated-k smoke
records are in the common contract.  The historical PICASO 3.2.2 interpreter
remains unchanged for Version 1 and is forbidden for Version-2 molecular work.

## Frozen cases and gates

Every resolution uses an identical pressure-by-wavelength grey optical-depth
tensor at column optical depths `0`, `1e-6`, `0.1`, `1`, and `100`.  Each case
is run with a zero-emission lower boundary and a `1755 K` blackbody lower
boundary.  Optical depth is uniform per equal-log-pressure cell.  The upper
boundary has zero incident thermal intensity, flux is positive outward, and
the exact `6550 K` stellar blackbody normalizes eclipse depth with the frozen
WASP-17 area ratio.

Eight-point Gauss-Legendre quadrature in emission-angle cosine uses normalized
`2 mu` disk weights.  The exact continuous-angle no-bottom solution uses
`1 - 2 E3(tau)`.  Analytically exact blackbody signals and the zero-tau,
no-bottom flux are set to exact zero in diagnostic signal arrays; raw solver
fluxes and cancellation values remain preserved.

The complete arrays were not inspected before the following code-level gates
were fixed:

| Diagnostic | Limit | Observed | Result |
| --- | ---: | ---: | --- |
| Analytic maximum symmetric relative difference | `5e-4` | `3.648947e-4` | pass |
| Analytic maximum eclipse difference | `0.01 ppm` | `0.196897 ppm` | **fail** |
| Compatible ROBERT/pRT maximum symmetric relative difference | `5e-5` | `1.65410e-6` | pass |
| Compatible ROBERT/pRT maximum eclipse difference | `0.01 ppm` | `0.00053724 ppm` | pass |
| Eight-angle versus continuous-angle symmetric difference | `5e-4` | `3.648947e-4` | pass |
| Maximum vertical-convergence eclipse difference | `0.01 ppm` | `7.0061e-12 ppm` | pass |
| Analytically handled blackbody signal | `1e-10 ppm` | `0 ppm` | pass |
| Zero-tau/no-bottom flux | `0 W m-2 m-1` | `0 W m-2 m-1` | pass |
| Maximum absolute single-scattering albedo | `0` | `0` | pass |

The sole failure is attributable to the declared finite angular
representation.  The relative angular discrepancy satisfies its own frozen
gate, but at the WASP-17 eclipse normalization it exceeds the separately
frozen analytic ppm gate.  This result is retained rather than increasing the
angle count, weakening the ppm threshold, or relabelling the continuous-angle
quantity after inspection.  Later claims requiring sub-`0.01 ppm`
continuous-angle agreement cannot use this eight-angle result as validated.

## Framework boundaries

ROBERT consumes both declared bottom boundaries directly through its
pure-absorption integrator.  Stable petitRADTRANS `fcore.compute_ck_flux`
consumes the identical tensor with its thermal blackbody lower boundary; it
does not expose the matching no-bottom convention, so those five arrays are
explicit NaNs under the artifact's unsupported-case policy and are not gated.

The PICASO 4.0 exact-`omega0=0` pilot probe is non-finite.  Stage 1 does not
replace zero with `1e-10`.  Instead, it retains an independently implemented
absorbing formal reference executed in the isolated PICASO process under the
label `picaso_absorbing_formal_reference`.  This reference checks array and
unit exchange but is not described or gated as native PICASO RT.

## Resources, artifacts, and warnings

The repeated final 80-cell pilot measured `13.329 s`, projected `46.653 s` for
the complete ladder, and measured a largest process peak RSS of `704,790,528
bytes` against `9,697,067,008 bytes` available (`7.27%`). It authorized the
run. The additional 40/160 matrix took `8.246 s`; total measured solver wall
time was `21.575 s`.

The versioned Stage-1 array artifact is approximately `74 MB`. Gate-bearing
native/R=100 flux, eclipse, optical-depth, and analytic arrays remain float64.
Complete vertical flux-contribution arrays are retained as documented float32
diagnostics to stay below GitHub's single-file limit.  Stable pRT unsupported
vertical/no-bottom products are explicit NaNs; all other arrays follow the
manifest's finite policy.

PICASO's known optional-Vega warning is retained in the environment contract;
no stellar grids are downloaded because Version 2 uses an exact blackbody.
The exact interpreters were:

```text
/opt/miniconda3/envs/robert-exoplanets/bin/python
/opt/miniconda3/envs/picaso-v4/bin/python
/opt/miniconda3/envs/petitradtrans-stable/bin/python
```

`stage_1_integrity.json` records checksums, shapes, dtypes, units, axes, and
finite policies.  `checksums.json` covers the committed Version-2 products.
Raw process contracts and worker outputs remain ignored beneath
`examples/outputs/emission_intercomparison/version_2/stage_1/`.

## Reproduction

Run from the repository root without changing environments:

```bash
PYTHONPATH=src /opt/miniconda3/envs/robert-exoplanets/bin/python \
  examples/benchmark_emission_intercomparison_v2_stage_1.py
```

The launcher runs the 80-cell resource pilot before the complete ladder and
preserves a scientific gate failure as a completed result rather than an
orchestration failure.
