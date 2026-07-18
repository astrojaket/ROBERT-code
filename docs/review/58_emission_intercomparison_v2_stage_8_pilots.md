# Emission intercomparison Version 2 Stage 8 controlled-study review

## Completed Stage-8 decision

Stage 8 is now a controlled common-denominator study of an idealized grey
aerosol. It asks how much enabling isotropic scattering changes each native
thermal-emission spectrum for the same accepted moderate Stage-7 placement.
The cloud top is `10 mbar`, the bottom is `100 bar`, column optical depth is
`tau=1` at `5 micron`, the extinction slope is zero, and `g=0` exactly.

Each framework runs clear (`tau=0`), absorbing (`tau=1`, exact `omega0=0`), and
scattering (`tau=1`, `omega0=0.9`) states. The current paths are ROBERT SH4/P3
with delta-M off, PICASO SH4 four-stream with delta-Eddington off, and stable-
pRT Feautrier isotropic scattering on the eight-angle grid. PG14 non-inverted
uses 40/80/160 cells and the isothermal control uses 80 cells: four shards,
three sequential states per shard, 12 cases per framework, and 36 native cases
total. The primary metric is scattering minus absorbing. No common cloud
tensor and no material-specific cloud claim are permitted.

## Controlled-study timing projection

The projection uses the completed representative pilots and freezes
`1.25 * (max(cold-warm,0) + 4*warm + 8*warm_case)`. The factor 1.25 includes
plotting, serialization, integrity, and report assembly. Each shard writes and
releases complete case diagnostics before the next state, retaining only its
small spectra concurrently.

| framework/path | projected wall (s) | projected peak RSS (bytes) | pilot available memory (bytes) | fraction available | strict 60% gate |
|---|---:|---:|---:|---:|---|
| ROBERT SH4/P3 | 198.39534291997552 | 6680389376 | 11029168128 | 0.605702016550128 | marginal fail |
| PICASO SH4 | 41.645089528901735 | 991652176 | 11608113152 | 0.08542750772800185 | pass |
| stable-pRT Feautrier | 51.35379999395809 | 2265409296 | 11558584320 | 0.1959936643867473 | pass |
| total serial | 291.39423244283535 | 6680389376 | -- | -- | conditional |

The predicted end-to-end time is `4.856570540713922 minutes`; a practical
laptop reservation is ten minutes. No cluster is required. PICASO and pRT are
comfortably safe. ROBERT is feasible in absolute terms--its projected peak is
well below measured available memory--but exceeds the conservative 60% gate by
`0.5702016550128` percentage points. Run frameworks serially, never launch
concurrent ROBERT workers, close memory-heavy applications, and require at
least `11,133,982,294 bytes` available before a ROBERT shard. If that preflight
fails, pause rather than changing precision, grids, cloud definitions, or the
solver.

## Controlled-study result

The authorized serial production run completed all 36 cases with finite native
and R=100 spectra. It took `355.57203475001734 s` (`5.926200579166956 min`),
`64.17780230718199 s` or `22.024390039968324%` above the pilot projection but
within the ten-minute laptop reservation. Peak process-tree RSS was
`8,827,633,664 bytes`; the minimum available memory before any launch was
`9,221,341,184 bytes`. Four early ROBERT cases used the authorized conservative
memory-gate override. The highest-memory case was the completed 160-cell ROBERT
scattering state, launched with `13,887,717,376 bytes` available. The 80-cell
pilot therefore underestimated 160-cell peak RSS; actual values supersede the
planning estimate.

The three frameworks recover the same broad response: small positive
scattering increments below roughly `2 micron`, followed by predominantly
reduced emission across `3--12 micron`. At the primary 80-cell PG14
non-inverted case, scattering-minus-absorbing RMS/max-absolute amplitudes are
`22.442024961294436/78.99316629388737 ppm` for ROBERT,
`26.60814803586827/94.02158412255514 ppm` for PICASO, and
`18.810476201617178/67.77954334260791 ppm` for stable pRT.

ROBERT-to-PICASO and ROBERT-to-pRT primary scattering-increment RMS differences
are `4.571871905839793` and `4.594008218071281 ppm`; PICASO-to-pRT is
`8.763732198286476 ppm`. These native Track-B differences are descriptive, not
gated. ROBERT and PICASO increment convergence is strong: 80-to-160 p95
difference over pair peak is `0.0010985944737690672` and
`0.0020003050038269305`. Stable pRT is less converged in the difference signal
at `0.0729768731965258`, although its absolute scattering-spectrum p95
symmetric difference is `0.005361423989234749`. This is a reported numerical
limitation and prevents claiming tighter pRT scattering-increment convergence.

Optional-Vega warnings from PICASO are irrelevant because the study uses the
explicit Version-2 blackbody star. ROBERT's OpenMP deprecation notice is also
retained. Complete numerical products, plots, worker logs, the report, and the
integrity index remain ignored locally. The compact versioned result is
`docs/data/emission_intercomparison/version_2/stage_8_controlled_study_summary.json`.

## Future-study pilot scope and pre-timing freeze

The older broad work is retained as resource planning for a **future study** of
independent native-code comparison (Track B). It cannot produce the current
Stage-8 science matrix and contains no Track-A
identical-tensor mode. The Version-2 common contract and completed Stage-7
contract remain authoritative without modification: WASP-17, the 6550 K
blackbody, composition/MMW/CIA definitions, PG14 profiles, 40/80/160 cells
(80 primary), signed flux/eclipse convention, 0.3--12 micron domain, 369 R=100
bins, native wavelengths, and lower-edge closure are unchanged. PICASO uses
only 4.0 resort-rebin correlated-k with the Stage-3 absolute summed-line-VMR
restoration.

The following definitions were frozen before inspecting timing results:

- pilot: PG14 non-inverted, 80 cells, the accepted Stage-7 10 mbar, slope-zero,
  tau=1 placement;
- Stage 8B: omega0=(0.5, 0.9, 0.99), tau=(0.1, 1, 10, 100), with tau=0.1/1
  classified as accepted-moderate and tau=10/100 as unresolved stress cases;
- Stage 8C: g=(0.3, 0.6, 0.9), delta-M off/on, and Toon/two-stream versus
  SH4/P3 only where the installed framework genuinely supports it;
- Stage 8D: analytic wavelength dependence is distinct from shared physical
  Mie inputs and each framework's native parameterization;
- Stage 8E: only stable-pRT's genuine Feautrier 16- and 32-angle paths are
  timed; ROBERT and PICASO have explicit unsupported high-order boundaries;
- float64 science arrays, exact zeros, retained warnings/non-finite status,
  `time.perf_counter`, cold and warm isolated processes;
- full counts include the isothermal control and both PG14 profiles, all three
  cell grids, controls, plots, and report assembly;
- nine laptop shards per solver path (one profile/grid pair per shard), 25%
  plot/serialization/integrity/assembly time allowance, 7200 s per-subsection
  laptop wall threshold, 21600 s combined threshold, and 60% of available
  memory as the serial-shard RSS threshold.

The exact supported paths, unsupported boundaries, case counts, parameters,
formulae, and thresholds are executable in
`emission_intercomparison_v2_stage_8_pilots.py`. The wall projection is
`1.25 * (max(cold-warm, 0) + 9*warm_setup + full_cases*warm_case)`. The memory
projection is measured pilot peak RSS plus retained case tensors for the
largest `ceil(full_cases/9)` shard. Definitions are not adjusted after timing.

## Future-study timing results

All 22 supported paths completed cold/warm process launches. ROBERT two-stream
failed natively with a singular boundary system for the frozen isotropic and
anisotropic cases. PICASO Toon returned non-finite spectra for its frozen
isotropic and delta-Eddington variants. Exact zeros, warnings, exception text,
time-to-failure, and RSS are retained; their wall projections are lower bounds,
not estimates of a successful matrix.

| subsection | projected serial wall (s) | serial peak RSS (bytes) | all-path parallel RSS (bytes) | paths |
|---|---:|---:|---:|---:|
| 8A | 130.40073303127429 | 4247581568 | 7280320048 | 3 |
| 8B | 2789.1672392569308 | 8156749184 | 18248981112 | 5 |
| 8C | 1731.1328952894837 | 7526665728 | 22860838016 | 7 |
| 8D | 1174.519274347549 | 7597844224 | 18769140912 | 5 |
| 8E | 1071.6494522131688 | 3217234960 | 5798000672 | 2 |
| combined | 6896.869594138407 | 8156749184 | 72957280760 | 22 |

All subsection serial wall projections are below the frozen two-hour threshold
and the combined serial projection is below six hours. Five ROBERT SH4 paths
exceed the frozen 60%-of-available-memory laptop rule (8B SH4/P3; 8C SH4
delta-M off/on; 8D spectral HG and physical Mie). Therefore a production run
must use one profile/grid shard at a time on a suitably provisioned node, not
all paths concurrently. A 16 GiB laptop is suitable for the external-framework
shards but is not accepted for those ROBERT SH4 shards under the frozen margin;
use at least a 32 GiB node. Running every supported path concurrently projects
72,997,372,408 bytes, so a 96 GiB cluster node is the minimum practical
single-node request; separate 32 GiB ROBERT, 16 GiB PICASO, and 16 GiB pRT
queues are preferred.

Because five Toon/two-stream rows use time-to-native-failure, the 8B, 8C, and
combined wall totals are lower bounds. They are valid resource-planning floors,
not predictions that those matrices can complete without resolving the native
numerical failures.

Large spectra, tensors, logs, raw timing products, checksums, and the local
integrity index remain ignored under
`examples/outputs/emission_intercomparison/version_2/stage_8/pilots/`. The
local `pilot_report.json` holds full-precision cold/warm wall, RSS, available
memory, setup/case time, native wavelength/bin/g/angle/stream counts, retained
tensor bytes, package versions, projections, warnings, and finite status.

## Future-study MgSiO3 scope

The material-specific comparison is deferred: MgSiO3 only, with cloud
parameterizations varying according to what each native code genuinely
supports. This scope is not retroactively used to tune the frozen pilots. The
future-study Stage-8D shared-physical pilot was already glassy
MgSiO3 using the measured Dorschner et al. optical constants at
`data/optical_constants/exo_skryer/MgSiO3.txt` (SHA-256
`0faef18dbd1ae853ef25be2f2766f499f9cb37d0732e0f401ca86e999c30c731`).

For a later production contract, shared information is restricted to the
MgSiO3 material identity and optical-constant provenance. ROBERT, PICASO/Virga,
and any supported pRT path must independently construct cloud optics through
their native parameterization. Radius/width, vertical distribution, optical
depth or condensate normalization, phase closure, delta-M behavior, and solver
order must be recorded per framework. A common phase, opacity, tau, or source
tensor is forbidden. The generic 8B/8C ladders remain solver controls and must
not be presented as predictions of MgSiO3 microphysics. The installed stable-pRT
data tree currently lacks MgSiO3 cloud-opacity assets, so its native
microphysical MgSiO3 path remains unsupported rather than being replaced by an
isotropic callback.
