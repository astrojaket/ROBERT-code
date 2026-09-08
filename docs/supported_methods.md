# Supported methods and validation status

This page is the current summary of the methods that ROBERT can run and the
evidence that supports each method. Use [forward-model generation](forward_models.md),
[running retrievals](retrievals.md), and [configuration](configuration.md) for
commands and YAML fields.

The labels below keep implementation status separate from scientific scope:

- **implemented** means that the package has the method and its interface;
- **tested** means that automated tests cover the stated behaviour;
- **cross-framework validated** means that a declared comparison with another
  solver or data source passed its stated checks; and
- **science demonstrated** requires a documented retrieval with data,
  posterior, and provenance checks. No general claim follows from an
  implementation or a unit test alone.

## Supported routes

| Method | Current route | Validation boundary |
|---|---|---|
| Clear thermal emission | Schema-v2 YAML, correlated-k gas opacity, CIA, and the absorbing formal solution | Stable petitRADTRANS 3.3.3 comparisons support the declared clear cases; they do not validate arbitrary opacity databases or clouds. |
| Cloudy thermal emission | Deck, haze, or Mie properties with Toon or thermal SH4/P3 | Matched PICASO/Virga and molecular-opacity comparisons cover declared closures; a high-stream reference is still required for a general cloud-science claim. |
| Transmission | Spherical shell chords with correlated-k or dedicated LBL input | Absorption-dominated and continuous reference comparisons are covered; returned scattered light, multiple scattering, and refraction remain outside the supported route. |
| Free and equilibrium chemistry | `FreeChemistry`, optional FastChem, and explicit composition/MMW models | FastChem and free-chemistry checks cover their configured inputs. They do not establish a complete chemical network or photochemical model. |
| Nested retrieval | PyMultiNest through the configured runner | R=100 injection checks and the documented target records cover validation-scale runs. Posterior calibration across broad seeds and all target regimes remains open. |
| Optimal estimation | Independent Gaussian observations through the configured runner | It rejects CLR and unsupported likelihood structures. Covariance and profiled high-resolution objectives need an explicit future residual/Jacobian contract. |

## Chemistry boundaries

Centered-log-ratio (CLR) composition priors use one joint unit-cube transform
for direct nested sampling. The derived background category is included in the
unit-sum composition. CLR priors are rejected for optimal estimation and hybrid
retrievals because their scalar density is not a valid replacement for the joint
transform. See the [chemistry guide](theory/chemistry.md).

A phantom background gas fills the remaining VMR and has no line, CIA, or
Rayleigh opacity. Its fitted molecular mass changes the mean molecular weight
and hydrostatic scale height. It is a composition category, not a hidden
opacity species.

Pressure quenching is an analytic decorator around a base chemistry model. The
implemented rule keeps the base profile below the quench pressure and uses the
equilibrium value at the quench pressure above it. Off-grid values use linear
VMR interpolation in `log10(P / bar)`, which is an explicit ROBERT convention.
The method is **tested**, but it is not **cross-framework validated** because no
authoritative NemesisPy profile oracle is available. No Taylor et al. retrieval
reproduction has been run. The [oracle contract](validation/pressure_quench.md)
defines what evidence is needed before either claim changes.

## Stellar contamination

The transit light source effect (TSLE) is a typed transform applied to the
planet-only transit-depth array on each configured observation dataset grid.
The implemented one- and two-heterogeneity transform preserves the
POSEIDON/Rackham arithmetic for the tested immaculate-chord cases. The compact
fixture and the [stellar-contamination theory note](theory/stellar_contamination.md)
define this dataset-grid parity boundary. It does not establish a general
high-resolution ordering relative to every instrument response or binning
operation.

PHOENIX stellar spectra are supported when the external STScI Synphot data are
installed. The POSEIDON parity result uses prepared synthetic spectra and does
not validate PHOENIX interpolation, active-region evolution, limb darkening,
spot crossings, or arbitrary stellar maps. A blackbody remains an explicit
controlled approximation.

## Line-by-line and high-resolution work

The LBL/HRS path keeps physical wavelength samples and applies the declared
velocity, line-spread, pixel, filter, and covariance operators. The current
release record approves native stride 1 for the measured gate and rejects the
tested stride-2 candidate. The [method review](review/51_line_by_line_high_resolution_retrieval_methods.md)
and [release manifest](data/lbl_hrs_release_manifest_20260831.json) contain the
compact checksums and resource limits.

LBL is a dedicated provider and is not an `opacity.format` option in the
standard YAML runner. Calibrated pipeline ingestion, arbitrary cross-order
covariance, and a full real-data shared-atmosphere HRS/LRS sampler run remain
deferred. The measured release therefore supports the declared method contract,
not a literature reproduction or a general high-resolution science claim.

## Accelerator boundaries

The `auto` thermal-integration setting selects the Numba kernel when it is
available and falls back to the NumPy reference. The NumPy path remains the
readable correctness reference. Thermal SH4 and the exact gas-combination
kernels must retain their declared numerical checks when a backend changes.

JAX CPU, JAX Metal, and opacity-sampling accelerator paths are optional
experimental work. Their component and single-proposal checks do not establish
posterior equivalence, long-run memory safety, or Metal/CUDA hardware science
validation. The [JAX Metal review](review/48_jax_metal_backend.md),
[withdrawn cloud-free benchmark](review/49_cloud_free_jax_metal_phase1.md), and
[opacity-sampling benchmark](review/50_jax_opacity_sampling_accelerator.md)
record the measured limits. Use the maintained Numba/NumPy CPU route for
production retrievals unless a separate release record approves another
backend.

## Evidence map

The maintained compact evidence includes:

- clear emission and absorption-dominated transmission comparisons with stable
  petitRADTRANS ([emission](review/20_petitradtrans3_multispecies_emission.md),
  [transmission](review/19_petitradtrans3_multispecies_transmission.md));
- cloudy emission parity and independent molecular-opacity checks
  ([matched cloud](review/35_end_to_end_picaso_virga_cloud_parity.md),
  [molecular cloud](review/36_official_picaso_molecular_cloud_parity.md));
- continuous transmission, stellar, and final molecular-opacity diagnostics
  ([continuous cloud](review/43_continuous_pt_cloud_reference.md),
  [POSEIDON transform](theory/stellar_contamination.md),
  [opacity attribution](review/45_final_molecular_opacity_attribution.md)); and
- LBL/HRS and target records referenced by the [data policy](data_policy.md).

These records are evidence for their named cases. They do not close the
unsupported regimes listed above.
