# Solar-composition Virga cloud forward sweep

**Result:** All 567 grid states are accounted for. There are 549
calculated cloud states and 18 explicit Virga domain rejections. The run
saved 1,098 cloudy spectra and six matched clear spectra, with no program or
worker failures. Forward-model time was 491.08 seconds on three local workers.
The strongest raw effects are near 1-1.3 microns in transmission and 9-12
microns in emission. Some extrema reach the model-top or particle-size limits;
their amplitudes need convergence tests before science use.

## Protocol

This controlled forward-model study uses ROBERT commit
`9ae7ca287199a1d2a5084143d2db69d4325e6081` and Virga 2.0.2 in the
`robert-exoplanets` environment. The protocol hash is
`2dab44cb06830dd67f66b03b6620ede013eba648b32074c719150b671919b834`.

FastChem calculates the full equilibrium composition for three prescribed
temperature-pressure profiles. Each profile contains 523 species. The
abundance reference is Asplund et al. (2009), with solar metallicity
([M/H] = 0) and C/O = 0.5495408739. Fifteen gases supply opacity:
H2O, CO, CO2, CH4, NH3, HCN, H2S, Na, K, SiO, SO2, C2H2, PH3, TiO and VO.
The setup check found all 15 tables with their native 20 g points. The three
FastChem profiles close in VMR to within 1.3e-12.

| Profile | T at 1e-5 bar | T at 1e-2 bar | T at 1 bar | T at 100 bar |
| --- | ---: | ---: | ---: | ---: |
| Cool | 700 K | 1000 K | 1400 K | 2800 K |
| Reference | 1000 K | 1300 K | 1700 K | 2900 K |
| Hot | 1300 K | 1600 K | 2000 K | 3000 K |

Temperature is linearly interpolated in log pressure to 64 layers. The study
does not extrapolate outside the four specified pressure knots.

The native opacity resolution is R=1000 with 20 g points. ROBERT bins the
correlated-k opacity tables onto 300 logarithmic wavelength bins from 0.6 to
12 microns, then solves the spectra on that grid, at nominal R=100. The native
g ordinates are retained; their float32 header weights are normalized to unit
sum. CIA, Rayleigh and equilibrium H-minus continuum
are included. The reference planet has radius 80,000 km, mass 1.1502935e27 kg
and constant gravity 11.99594 m/s2, with the transmission radius defined at
1 bar. The reference star is a 5000 K blackbody with radius 600,000 km.

The seven cloud species are separate cases: MgSiO3, Mg2SiO4, Fe, Al2O3, MnS,
Na2S and KCl. Each case uses a Virga Mie table generated from the pinned
optical-constant catalogue. The tables cover 0.4–14 microns on 713 wavelength
points and the Virga default 60-radius grid (0.0001–542.391 microns). All seven
tables passed readback and physical-bound checks. They use the pinned
exo_skryer optical-constant catalogue at commit
`abde13e1e39753f0528e271227decec503211c64` (AGPL-3.0). The external input
manifest records source references and file hashes. Setup evidence is saved in
`shared-inputs/inputs_manifest.json`, `root_checks/setup15_verification.json`
and `root_checks/mie_verification.json` under the external artifact root below.

Cloud grids use fsed = 0.1, 0.3, 1, 3 or 10 and log10(Kzz / cm2 s-1) = 7, 8,
9, 10 or 11. The main particle-distribution width is geometric standard
deviation 2. At the fiducial point fsed=1 and log10(Kzz)=9, two more widths
(1.5 and 3) are tested. Each cloud species is tested alone. The requested grid
has 567 states across the three temperature profiles. It would produce 1,140
spectra if every cloud state were supported. The 18 rejected states have no
spectrum; they are not replaced with clear spectra.

Virga cloud-vapour abundances use Virga 2.0.2 solar `gas_properties` at each
profile's recorded Virga mean molecular weight. The study records physical
gas mass ratios in g/g. FastChem uses the Asplund 2009 elemental reference for
the atmospheric gas. These are separate abundance conventions. The cloud
cases do not combine condensates into a shared elemental budget. ROBERT keeps
the pressure-dependent FastChem mean molecular weight in its gas and
radiative-transfer calculations; the profile median mean molecular weight is
used for Virga only.

Emission uses thermal transfer with SH4 and a Henyey–Greenstein closure, plus
four-point disk integration. The same emission settings apply to clear and
cloudy cases. Transmission uses spherical-shell extinction, constant gravity
and six-point impact integration.

## Opacity provenance

The VO input is the official [ExoMol VOMYT R=1000 KTA](https://exomol.com/db/VO/51V-16O/VOMYT/51V-16O__VOMYT.R1000_0.3-50mu.ktable.NEMESIS.kta),
from the [VOMYT dataset page](https://exomol.com/data/molecules/VO/51V-16O/VOMYT/).
This is the 2016 VOMYT line list. The installed file is
`/Users/jaketaylor/Developer/ROBERT-code/opacity_data/ktables_exomol/VO_R1000.kta`,
243,275,760 bytes, SHA256
`0fcbe5a46bc55d0a1bf516caa6462b28a716e8e9c19eb8d64707ed89257aaf15`. Pinned
ROBERT readback checked its 20-g header and all 60,813,720 stored coefficients.
The file structure and coefficient values passed; this is not a validation of
the line list's scientific accuracy.

The newer 2024 HyVO HITEMP KTA was not used. Its downloaded file has the same
SHA256 as the old local VO table and fails pinned-reader header validation:
`irec0=5221` implies 5,116 wavelength records, but the header declares 5,119.
The old local file remains unchanged. No header repair was made. This defect
applies to this KTA artifact; it does not establish a broader judgment about
HyVO line-list science. The detailed header audit is in the external VO input
summary at `/Users/jaketaylor/.codex/visualizations/2026/11/06/01a1b0d4-f26e-7571-83c0-e1a6be3d8b9a/virga-jwst-parameter-sweep-20261107/shared-inputs/vo_download/VO_INPUT_SUMMARY.md`.

The dataset page gives the VOMYT line-list and ExoMolOP table references.
The saved input manifest retains the original source metadata.

Opacity files stay in ignored or external input directories. No heavy opacity
file is part of this documentation change. The external run and input root is
`/Users/jaketaylor/.codex/visualizations/2026/11/06/01a1b0d4-f26e-7571-83c0-e1a6be3d8b9a/virga-jwst-parameter-sweep-20261107`.

## Interpretation limits

The study compares forward spectra at prescribed temperature profiles and
solar gas composition. It does not solve radiative-convective equilibrium,
retrieve parameters, establish observational detectability or provide
posterior constraints. Cloud condensate depletion is not coupled back into
FastChem. Single-species cases avoid summing independent Virga solar
reservoirs, but they do not represent a jointly conserved multi-condensate
cloud. Cloud, gas, and temperature assumptions remain fixed as specified.

The Virga adapter uses an asymmetry-factor closure for scattering. The planned
SH4/Henyey–Greenstein setup is a controlled comparison, not general validation
of multiple-scattering accuracy or strongly forward-scattering clouds. Mie
table coverage and bounds show that the selected tables load and span the
requested grid; they do not validate cloud formation physics.
For the wider experimental-provider scope and limits, see
[the Virga provider note](56_virga_cloud_experiment.md).

## Results

### Grid accounting and checks

The canonical ledger has 567 unique rows: 549 complete and 18 marked
`VirgaInvalidProposalError`. Of the complete states, one is the compatible
pilot checkpoint and 548 were calculated in the full invocation. The native
manifest retains `complete_with_errors`; the errors are these explicit domain
rejections, with zero failed program rows and zero failed workers.

Every rejection is at log10(Kzz)=7 and particle width 2. Virga reports that it
cannot balance the convective fall speed because the required particles would
be smaller than gas atoms. The rejected settings are:

| T-P profile | Cloud | fsed values | Count |
| --- | --- | --- | ---: |
| Cool | Fe | 0.1, 0.3, 1, 3, 10 | 5 |
| Reference | Fe | 0.1, 0.3, 1, 3, 10 | 5 |
| Reference | Al2O3 | 3, 10 | 2 |
| Reference | MnS | 10 | 1 |
| Hot | Al2O3 | 0.1, 0.3, 1, 3, 10 | 5 |

The reference MgSiO3 pilot matches the public high-level emission and
transmission forward paths exactly. Zero-cloud controls match the same-SH4
clear controls exactly. A separate NumPy/SciPy check agrees with the Numba
emission backend to within 2.61e-9 ppm and leaves its four input products
unchanged. Gas and noncloud arrays reconstructed for this check match the
saved arrays exactly. These checks establish calculation parity for this
pilot; they do not establish general cloud-physics accuracy.

The study uses edges at 1e-5 and 100 bar. The external driver rebuilds the
public forward models on this exact saved edge grid. Configured-task pressure
limits otherwise denote centre endpoints and extend their outer edges beyond
those limits. The final driver and model-grid records are hashed in the run
manifest; the study does not extend its opacity or T-P domain.

### Timing

The full forward run took 491.08 seconds, from 04:06:43 to 04:14:53 UTC on
7 November 2026. It used three spawned workers with one numerical thread each.
Peak RSS was 1.96-1.98 GB per worker. The final shared-input setup took 21.24
seconds, including 4.94 seconds for opacity preparation with already prepared
numeric caches. VO downloads, initial Mie generation, pilot checks and plot
production are outside this forward-run timer.

| Stage | Sum across newly completed states (s) |
| --- | ---: |
| Virga cloud optics | 382.14 |
| Virga cloud profiles | 380.78 |
| Emission transfer | 171.87 |
| Transmission transfer | 16.83 |

These are concurrent worker-stage sums, not elapsed serial timings. The
optics and profile APIs each calculate a Virga state. Together they account
for about 80% of these four timed stages. Removing that duplicate calculation
is the first measured performance target. The pilot's median warm cloudy
emission solve took 0.289 s with Numba and 0.713 s with NumPy/SciPy (three
calls each). This 2.47x kernel comparison is not an end-to-end speedup claim.

### Spectral sensitivity

All differences below mean **cloudy minus the matched clear model**. Emission
uses absolute planet/star eclipse-depth differences in ppm. Transmission uses
absolute transit-depth differences in ppm. Emission is not divided by the
clear spectrum. These are deterministic grid ranges, not posterior intervals.

| Profile | Geometry | Raw peak change (ppm) | Wavelength (micron) | Cloud | fsed | log10 Kzz |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Cool | Transmission | +1155.716 | 1.034 | Al2O3 | 0.1 | 7 |
| Reference | Transmission | +1371.292 | 1.275 | Al2O3 | 0.1 | 7 |
| Hot | Transmission | +1177.830 | 1.288 | Al2O3 | 0.1 | 11 |
| Cool | Emission | -2176.593 | 9.681 | Mg2SiO4 | 0.1 | 7 |
| Reference | Emission | -3248.912 | 9.779 | Mg2SiO4 | 0.1 | 7 |
| Hot | Emission | -1060.864 | 3.941 | Al2O3 | 0.1 | 10 |

All six raw extrema have particle width 2. The four cool/reference extrema
include effective radii outside the Mie radius-centre range in some upper layers.
The two silicate emission extrema also have very large optical depth in the
top layer. Thus this table locates strong response regimes; it does not give
converged amplitude limits.

For a one-parameter comparison, the maximum spectral span is the largest
wavelength-wise difference between parameter values while the other controls
stay fixed. The four largest width-2 spans are all reference-profile Mg2SiO4:

| Varied control | Fixed control | Geometry | Peak span (ppm) | Peak wavelength (micron) |
| --- | --- | --- | ---: | ---: |
| fsed, 0.1-10 | log10 Kzz = 11 | Transmission | 1317.002 | 1.034 |
| fsed, 0.1-10 | log10 Kzz = 7 | Emission | 2951.682 | 9.681 |
| log10 Kzz, 7-11 | fsed = 0.1 | Transmission | 991.038 | 1.034 |
| log10 Kzz, 7-11 | fsed = 0.1 | Emission | 2065.013 | 9.779 |

Some endpoint models in these spans have the support flags described below.
The main response is strongest at low fsed. The conditional comparisons
measure Kzz response at fixed fsed. Particle width
also matters: varying it across 1.5, 2 and 3 at fsed=1 and log10 Kzz=9 gives a
largest peak span of 395.520 ppm for cool Mg2SiO4 emission. Its largest
band-mean span is 122.009 ppm for reference Fe emission.

Conditional tables record the available values and missing values for every
slice. Three fsed slices have no supported states; two have partial coverage.
Eighteen Kzz slices omit the unsupported value 7. Partial spans use the actual
available parameter width when a per-decade value is reported. A cell with
fewer than two supported values has no response estimate.

Condensation changes sharply with T-P profile. The cool profile forms all
tested species except KCl. The reference profile forms MgSiO3, Mg2SiO4, Fe,
Al2O3 and MnS. Only Al2O3 forms in the hot profile. KCl has zero condensate in
all three profiles; Na2S forms only in the cool profile. These null results
apply to these columns and solar reservoirs, not to the species in general.

### Particle and upper-boundary checks

There are 243 states with zero condensate and 306 with condensate. All 306
have condensate and positive cloud optical depth in the top layer. None has
condensate at the bottom boundary. Thirty-nine states have effective radii
outside the Mie radius-centre range. Fifty-seven have more than 1% of the
condensate-pressure-weighted area or mass distribution outside the actual
outer radius-bin edges. These groups overlap. The largest tail fractions are
0.9928 in area and 0.9666 in mass for reference MnS at fsed=0.1, log10 Kzz=7.
Those spectra cannot establish a well-sampled particle-opacity response.

A geometry-only check reconstructs the hydrostatic upper-edge radii from the
saved columns. The corresponding opaque-edge transit depths are 19313.79,
19791.86 and 20276.63 ppm for cool, reference and hot. Fourteen state spectra
come within 1 ppm of that bound at some wavelength; 23 come within 10 ppm.
The maximum numerical overshoot is only 3.5e-12 ppm. The cool/reference
silicate emission extrema also reach this transmission bound. This is a
model-boundary warning, not evidence that the real atmosphere has a hard top.

A separate diagnostic ranking keeps only the 469 states with effective
radii inside the Mie centre bounds and both weighted tails at or below 1%.
The strongest emission changes then shift to Fe: 1824.963 ppm for cool at
log10 Kzz=10 and 2021.354 ppm for reference at log10 Kzz=11, both at fsed=0.1
and 11.473 microns. The hot Al2O3 extrema stay unchanged. An additional
exploratory filter requiring maximum top-layer vertical cloud optical depth
below 0.01 leaves 455 states. It still gives reference Fe changes of
1873.324 ppm in emission at 11.473 microns and 1231.151 ppm in transmission
at 1.034 microns, at fsed=0.1 and log10 Kzz=10. This numerical filter is not a
convergence test, and a small vertical optical depth does not guarantee a small
slant optical depth.

The useful next wavelength tests are therefore 0.8-1.5 microns for
transmission, 8-12 microns for cool/reference cloud emission, and the
3-5 micron region for the hot Al2O3 case. These are response targets for
follow-up tests, not observational constraints or instrument forecasts.

### Saved-product and figure review

The independent saved-product verifier passes 9,150 checks with zero failures.
Root also compared all six raw peak metrics and all 420 conditional fsed/Kzz
spans against a separate calculation from the 549 saved state arrays. Every
span agrees exactly. No new forward call was made for analysis or these
comparisons; the extra upper-bound calculation evaluates hydrostatic geometry
only. The numerical and support checks do not substitute for physical grid
convergence.

Seven PNGs and seven rendered vector PDF pages pass the final visual review.
They use the WASP-15b style with external captions. The figures show T-P
profiles, fiducial condensate profiles, fiducial spectral differences, grid
maxima, and conditional fsed/Kzz/width responses. No sampler, noise model or
1-sigma/2-sigma probability band is used in this forward grid. Full arrays,
figures and logs remain outside Git at the recorded artifact root. The
[compact sweep record](../data/virga_jwst_cloud_sweep_20261107.json) stores
summary numbers, settings, file hashes and evidence paths.

## Concrete next steps

The [local follow-up](64_virga_grid_convergence_2026-11-07.md) tests pressure
and radius grids for selected cases. It also verifies exact results from the
single-call Virga path and measures its effect on posterior calculation time.

1. Obtain a consistent low-pressure gas-opacity set and repeat the
   top-sensitive cases with it. The local extension tests use explicit gas
   opacity assumptions above the original pressure floor.
2. Add particle-support diagnostics to the Virga workflow. Repeat the 57
   tail cases with suitable radius coverage and record states outside the
   fall-speed validity range.
3. Check emission angular/stream convergence for selected strong Fe,
   silicate and Al2O3 cases. Then run injection recovery over the selected
   wavelength windows with temperature, gases, reference radius and clouds
   free. Run those retrievals to native convergence without an iteration cap.
   This tests constraints and degeneracies that the forward grid cannot.
4. Obtain a valid current HyVO correlated-k artifact or validate a proper
   format conversion that retains all g ordinates. Keep VOMYT and both
   original files with their provenance, then repeat a controlled VO opacity
   comparison.
