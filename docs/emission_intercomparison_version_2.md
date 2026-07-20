# Emission intercomparison Version 2

## Status and purpose

This is the canonical specification for Version 2 of the ROBERT, PICASO, and
petitRADTRANS thermal-emission intercomparison.  It freezes the physical
system, atmospheric composition, temperature profiles, grids, spectral
products, and interpretation rules before any Version-2 production matrix is
run.

The purpose is to identify where the three frameworks agree and where their
predictions diverge in controlled regimes.  The purpose is **not** to make the
codes agree, to remove native implementation differences, or to require every
framework to support every regime.  A documented capability limitation,
including a limitation in petitRADTRANS, is an intercomparison result.  Shared
contracts are used only when they are scientifically meaningful and genuinely
implementable in every participating path.

Version 1 remains historical evidence.  Its reports and arrays must not be
overwritten or silently reinterpreted as Version 2.

## Version-1 preservation

The completed Version-1 implementation is the history ending at ROBERT commit
`f00e0616c7aae7d37e0badda295c189ead17dde1` on
`codex/emission-intercomparison`.  The immutable archive tag is
`emission-intercomparison-v1`.  Committed Version-1 reports and arrays remain
under `docs/data/emission_intercomparison/`; Version-2 products go under a
new `docs/data/emission_intercomparison/version_2/` namespace.  Large local
raw products are to be checksummed and retained separately rather than copied
into Git or moved during an active stage.

The untracked `emission-intercomparison-paper/` directory is a separate Git
repository with concurrent manuscript work.  No intercomparison task may
edit, stage, clean, archive, checksum recursively, or otherwise touch it.

## Frozen physical system: WASP-17

The benchmark models WASP-17b.  The adopted bulk parameters are a single,
internally coherent Southworth/TEPCat solution, while the Earth distance is
the Gaia value presented by the NASA Exoplanet Archive.  This avoids mixing
the archive's independently selected default rows for mass and radius.

| Quantity | Adopted value | Use |
| --- | ---: | --- |
| Planet mass, $M_p$ | `0.477 +/- 0.033 M_J = 9.05346e26 kg` | Gravity and provenance |
| Planet radius, $R_p$ | `1.932 +/- 0.053 R_J = 1.38122544e8 m` | Gravity and eclipse normalization |
| Stellar mass, $M_star$ | `1.286 +/- 0.079 M_sun = 2.55717242e30 kg` | System provenance |
| Stellar radius, $R_star$ | `1.583 +/- 0.041 R_sun = 1.10129310e9 m` | Eclipse normalization and irradiation geometry |
| Stellar effective temperature | `6550 +/- 100 K` | Blackbody stellar spectrum |
| Stellar metallicity | `[Fe/H] = -0.25 +/- 0.09` | Recorded only; not used to set benchmark VMRs |
| Semimajor axis, $a$ | `0.05135 +/- 0.00103 AU = 7.681850660445e9 m` | Irradiation geometry |
| Orbital eccentricity | `0` | Frozen orbit |
| Orbital period | `3.73548546 +/- 0.00000027 d` | Current TEPCat ephemeris; provenance and future observation timing |
| Distance from Earth | `405.908 +8.779/-8.421 pc = 1.252501215748e19 m` | Provenance and optional absolute-flux products |

The adopted source values and uncertainties are documented in the
[TEPCat WASP-17 system solution](https://www.astro.keele.ac.uk/jkt/tepcat/planets/WASP-017.html)
and the [NASA Exoplanet Archive WASP-17 overview](https://exoplanetarchive.ipac.caltech.edu/overview/WASP-17).
The benchmark stores the quoted source uncertainties, but it does not sample
them during Stages 1--8.

The Version-2 conversion constants are frozen as `G = 6.67430e-11
m3 kg-1 s-2`, `M_J = 1.898e27 kg`, `R_J = 7.1492e7 m`,
`M_sun = 1.98847e30 kg`, `R_sun = 6.957e8 m`,
`AU = 1.495978707e11 m`, and `pc = 3.085677581491367e16 m`.  The derived
frozen quantities are:

- surface gravity `g = G M_p / R_p^2 = 3.1673143851664745 m s-2`;
- radius ratio `R_p/R_star = 0.12541851392694642`;
- projected area ratio `(R_p/R_star)^2 = 0.015729803635643653`;
- zero-albedo, full-redistribution equilibrium-temperature check
  `T_star sqrt(R_star/(2a)) = 1753.657719 K` (consistent with the rounded
  published `1755 K` value);
- geometric substellar irradiation-temperature check
  `T_star sqrt(R_star/a) = 2480.046530 K`.

All stored source values, constants, unit conversions, and derived values must
be covered by a contract test.  A task must not substitute another archive
default without creating a new explicitly named system-contract version.

WASP-17b is a useful emission target because JWST measurements already span
the NIRISS/SOSS and MIRI/LRS regimes.  The Version-2 benchmark is not a fit to
those observations, but it is designed so its later retrieval machinery can
be checked against a well-studied system.  Relevant context includes the
[JWST/NIRISS emission spectrum](https://arxiv.org/abs/2410.08149) and the
[JWST/MIRI dayside-emission analysis](https://arxiv.org/abs/2410.08148).

## Frozen stellar spectrum and eclipse convention

The stellar surface spectrum is an exact blackbody at `T_star = 6550 K`:

```text
B_lambda(T) = (2 h c^2 / lambda^5) /
              (exp(h c / (lambda k_B T)) - 1)
```

The implementation must use one declared set of physical constants, stable
`expm1` evaluation, SI units internally, and explicit conversions for any
framework-native unit.  No PHOENIX, Kurucz, opacity, limb-darkening, or stellar
line spectrum may enter Version 2.

The primary planet/star observable is the signed eclipse depth

```text
D_lambda = (F_p,lambda / F_star,lambda) * (R_p / R_star)^2,
```

where both fluxes use a compatible surface-flux convention.  Distance from
Earth cancels in the ratio; it is retained for provenance and separately
labelled absolute-flux diagnostics.  Every framework must record whether its
native output is intensity, hemispheric flux, luminosity density, or an
observer flux before conversion.

## Frozen atmospheric grid

- Pressure edges span `1e-5` to `100 bar`, uniformly in log pressure.
- The convergence ladder is `40`, `80`, and `160` cells; `80` cells is primary.
- ROBERT uses cells, PICASO receives the matching pressure levels required by
  its interface, and pRT receives the ROBERT geometric cell centres as pressure
  nodes.
- Arrays retain orientation metadata and explicit edge/centre semantics.
- No framework may independently trim the pressure range without recording
  the missing domain as a native limitation.

## Frozen temperature profiles

Version 2 replaces the hand-constructed monotonic, inverted, and
retrieved-like profiles used in Version 1 with reproducible analytic
Parmentier--Guillot 2014-style profiles.  The same evaluated temperature array
is supplied to every framework; a framework does not independently recreate
the profile.

The three reference cases are:

| Case | `T_int` (K) | `kappa_IR` (m2 kg-1) | `gamma1` | `gamma2` | `alpha` | `T_irr` (K) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Isothermal analytic control | n/a | n/a | n/a | n/a | n/a | n/a (`T = 1755 K`) |
| PG14 non-inverted | `100` | `0.001` | `0.1` | `0.5` | `0.5` | `1500` |
| PG14 inverted | `100` | `0.001` | `0.1` | `10.0` | `0.5` | `1500` |

The PG14 optical-depth coordinate is
`tau = kappa_IR * pressure(Pa) / g`, with the frozen WASP-17b gravity.  The
profile implementation, exponential-integral method, pressure units, and
parameter meanings must be checksum/version recorded.  `T_irr = 1500 K` is a
frozen analytic-profile parameter chosen to keep the photospheric temperatures
in a useful WASP-17b regime; it is not the geometric substellar temperature or
the published equilibrium temperature.

The 80-cell arrays generated from the canonical ROBERT implementation are the
contract arrays.  Independent framework PG14 evaluations may be retained as a
diagnostic but may not replace them.  Stage 4 verifies monotonicity/inversion
shape and Stage 9 retrieves the exact generating parameters.  The isothermal
case remains an analytic no-signal control: diagnostics that are exactly zero
by construction are treated as zero, not normalized numerical cancellation
noise.

## Frozen solar-like composition

The reference atmosphere is a deliberately controlled solar-metallicity
hot-Jupiter mixture.  It is not an assertion that WASP-17b has exactly these
abundances, and it does not inherit the host star's measured `[Fe/H]`.

The four active molecular VMRs were generated once from the repository's
versioned FastChem/Asplund-2009 solar-abundance setup at `[M/H] = 0`,
`C/O = 0.55`, `T = 1500 K`, and `P = 0.1 bar`.  They are thereafter constants:

| Species | Frozen VMR |
| --- | ---: |
| H2O | `3.222572565623962e-4` |
| CO | `4.598708447732890e-4` |
| CO2 | `6.734289181697181e-8` |
| CH4 | `3.861237147673902e-8` |
| H2 | `0.8540314245518249` |
| He | `0.14518634139157618` |

H2 and He fill the exact remaining fraction in the solar-like ratio
`0.8547:0.1453`; the six VMRs therefore sum to exactly one in the serialized
contract.  With the declared molecular masses, the reference mean molecular
weight is `2.321438174776293 u`.

These VMRs are fixed with altitude, temperature profile, resolution, stage,
and framework.  FastChem is provenance for the numbers, not a runtime native
chemistry path.  Allowed deliberate departures are narrowly labelled:

- Stage 2 isolates one active molecule at its reference VMR while preserving
  the H2/He fill convention;
- Stage 6 perturbs one reference log-VMR at a time and explicitly renormalizes
  the declared background;
- Stage 9 retrieves parameters around an injection made from this exact truth.

No stage may use convenient round-number VMRs, recompute equilibrium chemistry
along a PG14 profile, or accept a framework's default chemistry as if it were
the shared contract.  Additional native species can be explored only in a
separately labelled attribution track.

## Spectral and opacity contract

- Primary comparison range: `0.3--12 micron`.
- Primary delivered grid: constant resolving power `R = 100`, with explicit
  edges and flux-conserving integration.
- Native-resolution wavelength, spectrum, opacity, and contribution/response
  arrays are retained alongside every R=100 product.
- The same wavelength convention (vacuum wavelength, increasing order) and
  spectral-density convention are recorded before comparisons.
- Common molecular and CIA source assets are versioned/checksummed where a
  genuinely shared opacity contract is possible.  Native opacity databases
  remain an attribution track rather than an agreement gate.

The exact endpoints define 369 equal-log R=100 bins. For correlated-k tables
whose first stored value is the centre of a bin bounded below by `0.3 micron`,
that first native value closes only the R=100 integration to the physical
lower edge. The stored native spectrum remains unchanged and no synthetic
point is labelled as native data.

PICASO opacity sampling is retired from the active Version-2 contract. It is
not run, plotted, gated, or used as an alternate molecular path. Historical
opacity-sampling products remain recoverable from Git history but are not
active artifacts under the `0.3--12 micron` contract.

### PICASO molecular correlated-k decision

From Stage 2 onward, the primary PICASO molecular representation is the
official PICASO-4 resort-rebin correlated-k family from Zenodo
`10.5281/zenodo.18644980`, Version 2.  The frozen H2O, CO, CO2, and CH4 asset
filenames, SHA-256 checksums, common 661-bin/eight-double-Gauss representation,
20-by-73 pressure-temperature coverage, and source provenance are serialized in
`version_2/common_contract.json`. PICASO resort-rebin correlated-k is the only
active PICASO molecular representation.

Version 2 uses the process-isolated PICASO 4.0 / Python 3.11.15 interpreter at
`/opt/miniconda3/envs/picaso-v4/bin/python`.  Its reference tree is
`/Users/jaketaylor/Dropbox/picaso-v4/reference`.  Workers set `picaso_refdata`
to that path and set writable task-local `NUMBA_CACHE_DIR` and `MPLCONFIGDIR`
before importing PICASO.  The four-molecule resort-rebin correlated-k smoke
passes over all 613 native bins in the frozen domain. The historical PICASO 3.2.2
interpreter remains unchanged for Version 1 and must not be used for Version-2
molecular work.

## Common comparison architecture

Where a stage permits it, use two tracks:

- **Track A -- shared numerical contract:** identical pressure-by-wavelength
  optical-depth and applicable scattering arrays are supplied to compatible
  RT paths.  Acceptance gates test only quantities with matched definitions.
- **Track B -- native framework construction:** each code builds its native
  opacity/cloud representation from the same high-level physical case.
  Differences in databases, interpolation, units, boundaries, phase moments,
  and unsupported capabilities are attribution results.  Gates are not
  invented merely to force native implementations to agree.

All gates must be written in code and documentation before the complete matrix
is inspected and must not be tuned afterward.  Failure does not automatically
block the next scientific stage: the failed regime remains visible, its use as
a validated dependency is restricted, and the project proceeds where the
question can still be asked honestly.

ROBERT, PICASO, and pRT always execute in separate processes using:

```text
ROBERT: /opt/miniconda3/envs/robert-exoplanets/bin/python
PICASO: /opt/miniconda3/envs/picaso-v4/bin/python
pRT:    /opt/miniconda3/envs/petitradtrans-stable/bin/python
```

Every stage records absolute interpreter paths, package and data versions,
known warnings, exact input/method contracts, source/data/output checksums,
random seeds, raw and summarized timings, and peak resident memory where
available.

## Version-2 stages and rerun decision

All completed Version-1 stages are rerun.  Even Stage 1 changes because the
WASP-17 radius ratio, stellar blackbody, common wavelength range, grid
contract, and Version-2 serialization alter the observable contract.

### Stage 1 -- grey/isothermal closure

Re-establish units, boundary conventions, angular integration, signed spectra,
and eclipse-depth normalization with the WASP-17 system and 1755 K isothermal
control.  Compare analytic pure-absorption solutions and identical grey
optical-depth tensors on 40/80/160 cells.  Build and integrity-test the common
Version-2 system, composition, PG14, pressure, wavelength, and stellar
contracts before later stages depend on them.

Stage 1 was completed on the 40/80/160-cell ladder.  Eight of nine frozen
scientific gates pass.  The one retained failure is the continuous-angle
analytic eclipse gate: the eight-angle quadrature differs from the exact E3
solution by at most `3.648947e-4` symmetrically, which passes its dedicated
`5e-4` angular gate, but corresponds to `0.196897 ppm` and exceeds the
independently frozen `0.01 ppm` analytic eclipse limit.  The limit was not
relaxed after the matrix was seen.  On the genuinely compatible shared-tensor,
blackbody-lower-boundary subset, ROBERT and stable pRT close to
`6.75018e-7` symmetric relative difference and `0.00053724 ppm`; 80-to-160
changes are at most `7.23e-12 ppm`.  PICASO's exact-`omega0=0` native probe is
non-finite, so its independently implemented absorbing formal reference is
retained under a separate label and is not presented as a native PICASO gate.
See `docs/review/51_emission_intercomparison_v2_stage_1.md`.

### Stage 2 -- single-molecule closure

Run H2O, CO, CO2, and CH4 separately at their frozen reference VMRs.  Compare
shared pressure-by-wavelength optical-depth inputs and native opacity paths,
retain native and R=100 spectra, and quantify vertical and spectral convergence.
Do not substitute `1e-4` or other round-number abundances.

Stage 2 is complete.  The frozen matched ROBERT/pRT Track-A limits identify an
out-of-tolerance, vertically converging closure regime: maximum eclipse-depth
differences decrease from `10.150094 ppm` at 40 cells to `2.558145 ppm` at 80
and `0.640615 ppm` at 160. The isothermal analytic controls remain below
`0.001044 ppm`. Track B uses PICASO resort-rebin correlated-k only. See
`docs/review/52_emission_intercomparison_v2_stage_2.md` and
`docs/data/emission_intercomparison/version_2/stage_2_report.json`.

### Stage 3 -- multi-species and CIA

Combine the exact six-species mixture, then add H2--H2 and H2--He CIA in a
factorial design.  Separate line-opacity, CIA, mean-molecular-weight, and native
database effects.  Use the PG14 non-inverted profile as the headline physical
case and retain the isothermal analytic control.

Stage 3 is complete.  H2O, CO, CO2, and CH4 remain simultaneously active at
their exact frozen solar-derived common-contract abundances in every case, while H2--H2 and
H2--He CIA form a `2 x 2` on/off factorial.  The frozen `1755 K` isothermal and
PG14 non-inverted arrays were supplied identically on 40, 80, and 160 cells,
with 80 cells primary.  The representative 80-cell line-plus-both-CIA pilot
measured `12.215756 s`, projected `439.767203 s`, and used at most
`4,149,035,008 bytes`, or `34.18%` of available memory, authorizing the full
matrix. The post-pilot matrix took `166.935993 s`.

The matched ROBERT/stable-pRT Track-A limits again identify an
out-of-tolerance, vertically converging closure regime rather than a framework
failure. Maximum eclipse-depth differences decrease from `7.110565 ppm` at 40
cells to `1.788100 ppm` at 80 and `0.447547 ppm` at 160; the largest Track-A
80-to-160 change is `0.769349 ppm`. The isothermal analytic controls remain
below `0.001044 ppm`, and exact-zero scattering is preserved. These results do
not alter Stage 1's `0.196897 ppm` eight-angle result or permit sub-`0.01 ppm`
continuous-angle claims, and they do not reinterpret Stage 2's measured
out-of-tolerance vertical regime.

Track B is native database/interpolation/representation attribution only and
has no cross-framework gate.  At 80 cells the maximum native R=100 difference
is `6.499018 ppm` between ROBERT and stable pRT and at most `74.539002 ppm` for
pairs involving PICASO correlated-k. The PG14 both-CIA versus molecular-only
effect is `43.387801 ppm` in ROBERT, `43.656935 ppm` in stable pRT, and
`45.501171 ppm` in PICASO after restoring the absolute four-molecule VMR in
PICASO's normalized resort-rebin mixer. PICASO's exact-`omega0=0` native probes and vertical arrays
remain capability evidence and absorbing-formal diagnostics respectively;
stable pRT still exposes no supported native layer optical-depth tensor.  See
`docs/review/53_emission_intercomparison_v2_stage_3.md` and
`docs/data/emission_intercomparison/version_2/stage_3_report.json`.

### Stage 4 -- PG14 structures and contribution functions

Compare the isothermal, PG14 non-inverted, and PG14 inverted contract arrays.
Report signed R=100 and native spectra/eclipse depths, complete normalized
vertical contribution functions where definitions permit, pressure centroids,
peak pressures, band/window behaviour, and 40/80/160 convergence.  Framework-
native diagnostic definitions remain separately named; similar-looking
contribution functions are not assumed identical.

Stage 4 is complete using the fixed Stage-3 molecular-plus-H2--H2/H2--He CIA
state. The exact common-contract isothermal, PG14 non-inverted, and PG14
inverted arrays were supplied identically at 40, 80, and 160 cells. The
initial cold-cache 80-cell three-framework pilot measured `24.917126 s`,
projected `448.508267 s`, and used at most `4,127,260,672 bytes`, or `38.72%`
of available memory, authorizing the matrix. The final reporting run retained
a repeated `11.397582 s` pilot and `92.041606 s` post-pilot matrix timing.

The matched ROBERT/stable-pRT Track-A spectra remain an out-of-tolerance,
vertically converging closure regime. Maximum eclipse differences are
`7.110565`, `1.788100`, and `0.447547 ppm` at 40/80/160 cells. The inverted
profile has maximum symmetric relative differences of `1.865313e-2`,
`4.717567e-3`, and `1.182411e-3`; the largest Track-A 80-to-160 change is
`1.206041 ppm`. The isothermal analytic controls remain below `0.001044 ppm`.
The matched Track-A vertical diagnostic agrees exactly to stored precision.

Track B remains ungated native attribution. At 80 cells, the largest native
ROBERT/stable-pRT difference is `9.873270 ppm`, and the largest pair involving
PICASO correlated-k is `125.715939 ppm`, both for the inverted profile. The
largest native pressure-centroid RMS difference is `0.129370 dex`; complete
pressure-resolved diagnostics, centroids, peaks, seven optical-to-mid-IR
band/window summaries, and convergence arrays are retained. PICASO's native
total `taugas` and exact-`omega0=0` thermal probe remain separate from the
absorbing-formal spectrum and vertical diagnostic. Stable pRT's supported
native flux path still exposes no layer optical-depth tensor. See
`docs/review/54_emission_intercomparison_v2_stage_4.md` and
`docs/data/emission_intercomparison/version_2/stage_4_report.json`.

### Stage 5 -- temperature responses

Stage 5 is complete. It applied the same six Gaussian log-pressure
localizations at `1e-4` through `10 bar` with `0.35 dex` width and symmetric
`+/-10 K` amplitude to the exact three Stage-4 profiles. A predeclared
`+/-5/10/20 K` primary-grid ladder verified finite-difference linearity and
symmetry. Exact-zero normalized responses remain exact zero without epsilons.

Track A freezes the completed Stage-4 shared mean molecular-plus-both-CIA
optical depth while perturbing the source in the compatible ROBERT and stable-
pRT paths. All frozen gates pass: the primary Jacobian p95 difference is
`0.00168817`, the primary eclipse-Jacobian RMS difference is
`0.000786942 ppm/K`, the primary response-centroid RMS difference is
`0.00174522 dex`, and response TV p95 is `0.00252234`. The corresponding
80-to-160 values are `0.00151578`, `0.000648841 ppm/K`, `0.000829095 dex`,
and `0.00207203`. The isothermal analytic control is `0.001043488 ppm`.

Track B recomputes native opacity for every required perturbed temperature
state and remains attribution-only. At 80 cells ROBERT/stable-pRT Jacobian p95
differences reach `0.515848%`; pairs involving PICASO reach `4.05137%`.
Complete signed native/R=100 states, response/Jacobian tensors, zero-signal
masks, localization arrays, centroids, peaks, band/window diagnostics,
linearity arrays, supported opacity tensors, and convergence are retained in
154 sharded Stage-5 products. Contribution functions and temperature
responses are compared but explicitly not treated as identical. See
`docs/review/55_emission_intercomparison_v2_stage_5.md` and
`docs/data/emission_intercomparison/version_2/stage_5_report.json`.

### Stage 6 -- composition responses

Stage 6 is complete. Each of H2O, CO, CO2, and CH4 was perturbed independently
with symmetric localized log10-VMR changes at the same six pressure centres,
using `0.35 dex` Gaussian width and a primary `+/-0.10 dex` half-step. The
exact Version-2 `0.8547:0.1453` H2/He remainder, complete six-species
normalization, MMW, H2--H2 CIA, and H2--He CIA were recomputed for every state.
The `0.05/0.10/0.20 dex` ladder was evaluated on both 80-cell PG14 profiles.

Track A recomputes the shared molecular-plus-both-CIA optical depth for every
composition state and is limited to compatible ROBERT/stable-pRT definitions.
All primary and 80-to-160 cross-framework gates pass: primary signed-Jacobian
p95 is `0.326129%`, eclipse-Jacobian RMS is `0.132753 ppm/dex`, pressure-
centroid RMS is `0.00197857 dex`, response TV p95 is `0.00301641`, and cross-
species-fraction TV p95 is `0.000157077`. The corresponding converged values
are `0.215698%`, `0.0747847 ppm/dex`, `0.00894567 dex`, `0.0167956`, and
`0.0000558249`.

The `0.00217799` linearity p95 passes, while the `0.0266245` symmetry p95
exceeds its frozen `0.02` gate. Stage 6 is therefore retained as an out-of-
tolerance characterized regime, not a framework failure. The isothermal
analytic composition response and all zero-signal normalizations are exactly
zero.

Track B remains attribution-only and recomputes native molecular opacity,
mixing, CIA, composition conversion, and MMW for each state. Primary native
ROBERT/stable-pRT Jacobian differences reach `0.815618%`; pairs involving
PICASO correlated-k reach `4.071860%`. PICASO's state-dependent absolute
summed-line-VMR restoration is verified per perturbation. Complete signed
state spectra, composition responses/Jacobians, own/cross-species fractions,
pressure diagnostics, supported optical-depth tensors, band/window summaries,
and convergence were retained in 767 locally generated checksum-indexed
products. They default to the ignored Stage-6 output tree for later Zenodo or
equivalent archival release rather than ordinary Git. See
`docs/review/56_emission_intercomparison_v2_stage_6.md`.

### Stage 7 -- absorbing-cloud placement and extinction

The contract, implementation, and complete matrix are finished. The mandatory
frozen resource pilot initially stopped execution before matrix inspection. The
80-cell PG14 non-inverted clear/`tau=1`, 10-mbar, slope-0 pilot used
`4,272,816,128 bytes` peak process-tree RSS, or `45.333370%` of the
`9,425,321,984 bytes` available at the decision, and therefore passed the
frozen 60% memory limit. Its projected complete wall time was
`8128.054204 s` (`2.258 h`) against the frozen `7200 s` limit, so it failed the
wall-time gate and set `continue_full_matrix=false`. The user subsequently
authorized exceeding that wall-time projection. No threshold, definition,
precision, or required case changed; the failed initial decision remains in the
compact summary.

The frozen matrix remains the Version-1 cloud design evaluated against the
fixed Version-2 gas, system, and PG14 profiles: grey optical depths `0.1`, `1`,
`10`, and `100`; tops at `1`, `10`, and `100 mbar`; slopes `-4`, `-2`, `0`, and
`+2`; and the shared archived physical extinction case, all at exact
`omega0=0`. Track A is restricted to genuinely identical ROBERT/stable-pRT
gas-plus-cloud optical-depth inputs; Track B is native attribution, and no
PICASO identical-tensor gate is invented.

The complete launcher took `2371.303374 s`: `42.664949 s` for its repeated
pilot and `2328.638425 s` for the matrix and assembly. Complete-matrix peak
process-tree RSS was `5,195,988,992 bytes`. The repeated warmed pilot projected
`4906.158647 s` and passed both resource limits; both pilot records are retained.

Stage 7 is an out-of-tolerance characterized regime. Exact isothermal cloud
effect and maximum `omega0` remain exactly zero, and all primary contribution-
profile gates pass. However, primary Track-A absolute-spectrum p95 is
`0.327970825`, cloud-effect p95 is `0.0517204233`, and eclipse-effect RMS is
`6.21989875 ppm`; all exceed their frozen limits. The corresponding 80-to-160
values are `0.288476674`, `0.0540403749`, and `6.33828924 ppm`; cloud-response
TV p95 is `0.644274045`. The strongest discrepancies occur in extreme
high-tau, high-altitude or short-wavelength-weighted placements, not the
moderate pilot alone.

Raw workers, 92 integrity-indexed full-precision products, the full report,
and the integrity index remain ignored locally for a later archival release.
See
`docs/review/57_emission_intercomparison_v2_stage_7.md` and the compact
`docs/data/emission_intercomparison/version_2/stage_7_summary.json`.

### Stage 8 -- controlled isotropic grey-aerosol study

Stage 8 now asks one deliberately narrow question: for the same moderate grey
cloud, how much does enabling exact isotropic scattering change each code's
native thermal-emission spectrum? It is a Track-B comparison of the simplest
completed native scattering path in each framework, not a material-specific
MgSiO3 or microphysical-cloud claim.

The common analytic cloud has a `10 mbar` top, `100 bar` bottom, grey column
optical depth `tau=1` at `5 micron`, spectral slope zero, and exact `g=0`. Each
framework evaluates this definition on its own native pressure and wavelength
grid; no opacity, phase, or optical-depth tensor is shared. The three states
are clear (`tau=0`), absorbing (`tau=1`, exact `omega0=0`), and scattering
(`tau=1`, `omega0=0.9`). The primary diagnostic is scattering minus absorbing,
which separates the scattering response from cloud extinction.

The only production solver paths are ROBERT SH4/P3 with delta-M off, PICASO
SH4 four-stream with delta-Eddington off, and stable-pRT Feautrier isotropic
scattering on the eight-angle grid. ROBERT and PICASO Toon are excluded because
their frozen pilots failed or returned non-finite spectra. The matrix contains
the three states for PG14 non-inverted at 40/80/160 cells plus an 80-cell
isothermal control: 12 cases per framework and 36 native cases total.

The previously piloted isotropic ladders, anisotropy, delta-M, wavelength-
dependent/MgSiO3 clouds, native microphysics, and high-order reference envelope
are retained as **future-study** resource evidence. They are not part of the
current Stage-8 production contract and must not be needed to interpret this
controlled comparison.

The completed controlled run produced 36/36 finite cases in `355.572035 s`
(`5.926201 min`) with peak process-tree RSS `8,827,633,664 bytes`. All three
frameworks show the same broad scattering signature: small positive changes
below about `2 micron` and reduced emission through most of `3--12 micron`.
Primary scattering-minus-absorbing RMS amplitudes are `22.4420 ppm` (ROBERT),
`26.6081 ppm` (PICASO), and `18.8105 ppm` (stable pRT). Native pairwise RMS
differences are `4.5719--8.7637 ppm` and are descriptive, not gated. Stable
pRT's 80-to-160 scattering-increment p95 difference is `7.2977%` of pair peak,
larger than ROBERT/PICASO, so the controlled conclusion does not claim tighter
pRT difference-signal convergence. See the compact Stage-8 summary and
`docs/review/58_emission_intercomparison_v2_stage_8_pilots.md`.

### Stage 9 -- directed JWST-like cross-retrievals

Generate injections from the exact Version-2 contracts and perform all six
directed injection/retrieval code pairs, without self-retrievals.  The
cloud-free arm retrieves the two PG14 families, four molecular abundances, and
no area or radius normalization.  The cloudy arm uses only the grey regimes
supported by the Stage-7/8 maps and natively retrieves cloud optical depth and
cloud-top pressure; the isotropic-scattering case also retrieves single-
scattering albedo.

Following Barstow et al. (2020), use one unperturbed synthetic spectrum per
injector/scenario and do not add Gaussian random draws or self-retrievals.
Apply `30`, `60`, and `100 ppm` uncertainty envelopes in the Gaussian
likelihood, with frozen priors/transforms and explicit failure reporting. This
gives 72 science retrievals. MultiNest 3.10/PyMultiNest 2.12 runs at 400 live
points with 12 MPI ranks and one thread per rank on Glamdring's `redwood`
queue. Stage 9 is prepared and structurally tested locally; all native
injection, pilot, and retrieval science execution occurs only on Glamdring
after explicit approval. All active PICASO injections use the frozen
resort-rebin correlated-k representation.

The frozen setup is
`docs/data/emission_intercomparison/version_2/stage_9_retrieval_contract.json`;
operational details are recorded in
`docs/review/59_emission_intercomparison_v2_stage_9_setup.md`.

## Required products for every Version-2 stage

- a JSON report with schema version, frozen contract hashes, methods, gates,
  numerical summaries, warnings, timings, and resources;
- a descriptive versioned NPZ (or multiple sharded NPZs when justified) with
  every complete numerical array used by the report;
- an integrity manifest with SHA-256 checksums, keys, shapes, dtypes, units,
  axis meanings, and finite/non-finite policy;
- a review document that reports actual results and deviations without
  collapsing them to one ranking;
- focused contract/metric/artifact tests, Ruff, all three environment smoke
  tests, and the complete ROBERT pytest suite;
- a representative primary-resolution pilot before a large local matrix, with
  projected wall time and memory.  Stop before the full matrix if the estimate
  exceeds roughly two hours or lacks comfortable memory margin.

## Common preamble for every stage task

Every Version-2 stage or substage prompt must begin with the following block,
changing only the expected base commit and the assigned task branch/worktree:

```text
This task implements ROBERT emission intercomparison Version 2. Work only in
the Codex worktree assigned to this task, based on
codex/emission-intercomparison-v2 at <EXPECTED_BASE_COMMIT>. Before editing,
verify the current branch/base, HEAD, and clean status. Do not reset, merge, or
switch the primary checkout. If the expected base is not present, stop and
report the mismatch rather than rebuilding work from memory.

Read AGENTS.md and docs/emission_intercomparison_version_2.md completely, then
read the canonical roadmap, environment record, preceding Version-2 reviews,
and the implementations/tests relevant to this stage. The Version-2 document
is the controlling scientific contract. Do not change the frozen WASP-17
system, blackbody star, solar-like VMRs, PG14 arrays/parameters, 40/80/160
pressure grids, 0.3--12 micron comparison domain, or R=100 product definition
inside a stage. Propose any required contract revision separately before
running results.

The intercomparison describes agreement, deviations, and capability limits; it
does not modify or tune codes to force agreement. A missing or limited pRT,
PICASO, or ROBERT capability is an explicit result. Use Track A gates only for
genuinely shared numerical contracts. Treat Track-B native differences as
attribution results unless definitions are truly identical. Predeclare gates
before viewing the complete matrix and never tune them afterward.

Keep ROBERT, PICASO, and stable petitRADTRANS process-isolated with exactly:
- /opt/miniconda3/envs/robert-exoplanets/bin/python
- /opt/miniconda3/envs/picaso-v4/bin/python
- /opt/miniconda3/envs/petitradtrans-stable/bin/python

Every PICASO worker must also set, before importing PICASO:
- picaso_refdata=/Users/jaketaylor/Dropbox/picaso-v4/reference
- NUMBA_CACHE_DIR=<writable task temp>/picaso-v4-numba-cache
- MPLCONFIGDIR=<writable task temp>/picaso-v4-matplotlib

Never use or reactivate retired Dropbox Git metadata; Dropbox is only linked
opacity-data/rollback storage. The untracked emission-intercomparison-paper/
directory is a separate concurrent Git repository: do not read recursively,
edit, stage, clean, checksum, or otherwise touch it, and do not treat it as a
ROBERT worktree change. Preserve all Version-1 files and write Version-2 data
only beneath the declared version_2 namespaces.

Preserve native-resolution arrays as well as flux-conserving R=100 products.
Use only the frozen PICASO-4 resort-rebin correlated-k path for active
Version-2 molecular work. Do not regenerate opacity-sampling products.

Record interpreter/package/data versions, known warnings, exact contracts and
definitions, checksums, timings, and peak resident memory. Pilot any large
matrix at 80 cells and stop before the full run if projected local wall time is
over roughly two hours or memory margin is unsafe. Run focused tests, artifact
integrity checks, Ruff, all three environment smoke tests, and the complete
pytest suite. Commit and push verified changes over SSH to the assigned
codex/ branch. Do not create a pull request unless explicitly requested.
```

## Detailed launch prompt: Version-2 Stage 1

The new Stage-1 task receives the common preamble above followed by:

```text
Implement Version-2 Stage 1: WASP-17b grey/isothermal closure and common-contract
bootstrap.

First preserve provenance. Verify the Version-1 endpoint commit recorded in the
Version-2 document, confirm that the separate manuscript repository remains
untouched, and create a small machine-readable Version-1/Version-2 lineage
record. Do not copy, move, or hash recursively through the approximately large
local raw-output tree during this stage.

Implement one typed, serializable Version-2 common contract used by later
stages. It must contain and validate:
- the adopted WASP-17b planet/star/orbit/distance values, source uncertainties,
  physical constants, SI conversions, derived gravity, radius ratio, area
  ratio, and equilibrium/substellar temperature checks;
- an exact 6550 K blackbody stellar surface spectrum with stable Planck
  evaluation and explicit spectral-density units;
- pressure grids with 40/80/160 cells over 1e-5--100 bar, 80 cells primary,
  including edges, geometric centres, orientation, and the PICASO/pRT mapping;
- the 1755 K isothermal analytic profile and both frozen PG14 parameter sets,
  plus versioned evaluated arrays and implementation/method checksums;
- the exact six-species VMR mapping, sum-to-one/background-fill rule,
  provenance state, molecular masses, and mean molecular weight;
- the 0.3--12 micron flux-conserving R=100 grid and native-spectrum retention
  requirements;
- the PICASO correlated-k-only rule and retired opacity-sampling state.

Add focused tests for source-to-SI values, derived quantities, blackbody limits
and units, VMR sum/MMW, PG14 shape and reproducibility, pressure mappings,
R=100 edge/integral behaviour, schema round-trip, immutability, and checksums.
No later stage should have to restate these values in ad-hoc dictionaries.

Then run the grey/isothermal closure with scattering explicitly disabled and
the stellar blackbody used for eclipse normalization. Exercise analytic and
shared pressure-by-wavelength grey optical-depth cases sufficient to test zero,
optically thin, intermediate, and optically thick limits; angular quadrature,
top/bottom boundary conventions; signed surface flux and eclipse depth; native
and R=100 representations; and 40->80 and 80->160 convergence. Supply identical
optical-depth tensors to every compatible pure-absorption path. If a framework
cannot consume an identical tensor, document that boundary and retain its
native result without inventing a gate.

Treat analytically exact blackbody/no-signal cases explicitly as zero rather
than dividing by cancellation noise. Freeze all Stage-1 acceptance gates in
code and the review before inspecting the complete matrix. Preserve complete
spectral and vertical arrays, not only scalar maxima. Run a representative
80-cell three-framework pilot, record the measured resource projection, and
continue with the full Stage-1 matrix only if laptop-safe.

Write at minimum:
- docs/data/emission_intercomparison/version_2/common_contract.json
- docs/data/emission_intercomparison/version_2/version_2_common_profiles.npz
- docs/data/emission_intercomparison/version_2/stage_1_report.json
- docs/data/emission_intercomparison/version_2/stage_1_grey_isothermal_arrays.npz
- docs/review/51_emission_intercomparison_v2_stage_1.md

Update docs/emission_intercomparison_version_2.md,
docs/emission_intercomparison_roadmap.md,
docs/emission_intercomparison_environments.md, examples/BENCHMARKS.md, artifact
checksums, and tests as needed. Report actual numerical closure, convergence,
timing, memory, limitations, and warnings. End with a clean diff except for the
explicitly ignored separate manuscript repository, commit the verified work,
and push the assigned branch over SSH without opening a pull request.
```
