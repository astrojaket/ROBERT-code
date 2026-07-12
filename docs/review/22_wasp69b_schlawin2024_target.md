# WASP-69b Schlawin et al. (2024) Science Target

## Role in ROBERT validation

Schlawin et al. (2024), AJ 168, 104 is the first real-data end-to-end cloud and
inhomogeneous-dayside retrieval target for ROBERT. The published 2--12 micron
emission spectrum combines two JWST/NIRCam eclipses and one JWST/MIRI/LRS
eclipse. A homogeneous cloud-free model fails, while the favored explanations
require aerosols and either a cloud layer or an inhomogeneous two-region disk.

Primary references:

- Article: `https://doi.org/10.3847/1538-3881/ad58e0`
- arXiv: `https://arxiv.org/abs/2406.15543`
- VizieR: `J/AJ/168/104`
- MAST observations: `10.17909/v2v9-k243`
- Eureka control files: `10.5281/zenodo.11168833`

## Downloaded spectrum

The repository now contains the verbatim VizieR `ReadMe`, 280-point native
spectrum (`sp.dat`), and 79-point visualization spectrum (`spbin.dat`) under
`data/wasp69b_schlawin2024/`, with SHA-256 checksums and provenance.

The native dataset contains:

| Dataset | Points | Wavelength range |
|---|---:|---:|
| NIRCam/F322W2 | 144 | 2.455--3.884 micron |
| NIRCam overlap average | 6 | 3.894--3.944 micron |
| NIRCam/F444W | 102 | 3.954--4.964 micron |
| MIRI/LRS | 28 | 5.125--11.875 micron |

ROBERT preserves these as four named datasets rather than concatenating them
into an anonymous vector. Published bin widths are retained in provenance and
converted to contiguous bin edges after averaging only the tiny decimal-rounding
gaps in the machine-readable table.

## Inhomogeneous disk equations

The paper's two-region observable is

`F_obs = x_hot F_hot + (1 - x_hot) F_cold`,

where each column has its own temperature-pressure profile and gray cloud
opacity. Metallicity and C/O are horizontally shared. The retrieved hot area
fraction is approximately 0.68; the cold cloudy component contributes
negligible thermal flux in the median solution.

ROBERT implements this exactly in `TwoRegionEmissionModel`. Regional fluxes are
computed independently before projected-area mixing. The area fraction is
validated on `[0, 1]`; it is not implemented by averaging temperatures or
opacities, which would be physically incorrect because radiative transfer and
the Planck function are nonlinear.

These are not WASP-69b model classes. `DiskEmissionModelConfig` selects
`one_region`, `diluted_one_region`, or `two_region` and composes arbitrary
regional ROBERT emission evaluators. Each region continues to use the existing
`Planet`, `Star`, atmosphere builder, chemistry, opacity provider, and RT
interfaces. Target identity and system parameters therefore remain external
configuration, and the same disk choices apply independently of planet type.

The paper's dilution factor `s_dilute` is the fractional dayside area emitted
by one 1D model. ROBERT implements this separately in `DilutedEmissionModel` as
a multiplier on the final planet observable. It does not change radius,
temperature, opacity, gravity, or stellar flux. This is the limiting case of a
two-region disk when the complementary component has negligible flux.

## Multiple datasets and nuisance parameters

`ObservationCollection` and `ObservationDataset` preserve instrument identity,
binning, masks, and nuisance-parameter names. The default retrieval workflow
uses `exo_k` to recompress opacity into each instrument mode's published bins,
then evaluates one mode-specific forward model per dataset. This preserves the
correlated-k operation on opacity distributions before radiative transfer.

`MultiDatasetForwardModel` is retained for plotting and diagnostics. It
evaluates one shared native-resolution physical spectrum and integrates that
spectrum over each published top-hat wavelength bin; it does not substitute
centre interpolation. It must not be described as `exo_k` binning because
`exo_k` does not operate on an already evaluated flux spectrum.
`MultiDatasetGaussianLikelihood` sums independent likelihood terms while
applying offsets and jitter only to the named dataset.

The WASP-69b loader assigns the paper's optional `miri_offset` parameter only to
MIRI/LRS. Schlawin et al. explored a uniform prior from -500 to +500 ppm and
found bimodal solutions near -80 and +40 ppm, demonstrating that an anonymous
single offset for the combined spectrum would be scientifically wrong.

`MultiDatasetRetrievalProblem` now provides the sampler-independent parameter,
prior, prediction, likelihood, and posterior boundary required to retrieve
shared atmospheric parameters alongside per-dataset calibration parameters.

## Gray cloud implementation

The paper's CHIMERA two-region clouds are vertically uniform, gray, purely
isotropic scattering opacity, with independent hot/cold cloud strengths. Its
PICASO cloud-layer retrieval instead uses MgSiO3 Mie properties with a
log-normal particle-size distribution, cloud-base pressure, sedimentation
efficiency, vertical normalization, wavelength- and layer-dependent optical
depth, single-scattering albedo, and asymmetry.

`ParameterizedGreyCloudEmissionForwardModel` extends the existing runtime
temperature/chemistry regional model with a retrieval parameter
`log10(kappa_cloud / cm2 g-1)`. For a coefficient defined per gram of bulk
atmosphere, every layer uses the hydrostatic relation

`delta_tau_cloud = 0.1 kappa_cloud delta_pressure / gravity`,

where `0.1` converts cm2/g to m2/kg. The opacity is gray in wavelength and
uniform in mass extinction, not artificially divided equally among pressure
layers. Single-scattering albedo and asymmetry are explicit; the Schlawin
two-region case selects `omega=1`, `g=0`, with ROBERT's validated SH4 solver.
Independent hot and cold instances give independent cloud strengths without
introducing a planet-specific branch.

The staged ROBERT benchmark should therefore be:

1. reproduce the algebraic 2-region and dilution limits with synthetic spectra;
2. fit the real data with clear one-region, clear two-region, and diluted
   one-region models to establish the geometric evidence;
3. fit independent hot/cold gray isotropic-scattering clouds with SH4;
4. reproduce the paper's seven-parameter two-region cloud model;
5. add an MgSiO3 microphysical cloud provider and benchmark the cloud-layer
   scenario; and
6. compare retrieved metallicity, C/O, hot fraction, cloud contrast,
   contribution functions, likelihood, and evidence with the published values.

Cloud radiative feedback on the temperature profiles was omitted by the paper's
two-region retrieval and its reported uncertainties may therefore be too
narrow. ROBERT must record this as a model assumption and should later test a
self-consistent or broadened-profile sensitivity case rather than silently
inheriting the limitation.
