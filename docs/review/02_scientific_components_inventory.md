# Scientific Components Inventory

Sources inspected:

- NEMESIS / Radtran: https://github.com/nemesiscode/radtrancode at commit `52c273f`.
- NemesisPy: https://github.com/astrojaket/NemesisPy at commit `afe2bae`.

Recommendation vocabulary:

- Preserve concept: keep the scientific idea and expected behavior.
- Port with tests: rewrite in ROBERT with reference tests against NEMESIS/NemesisPy.
- Reuse temporarily: call existing code only as a validation bridge, not as permanent architecture.
- Redesign: keep only the scientific requirement, not implementation shape.

## Summary Matrix

| Component | NEMESIS maturity | NemesisPy maturity | ROBERT recommendation |
| --- | --- | --- | --- |
| Atmospheric state representation | High, legacy encoding | Medium, arrays/dicts | Redesign with typed immutable state objects. |
| Pressure grids and layering | High | Medium/high | Port with tests. |
| Hydrostatic height | High | Medium | Port with tests and unit checks. |
| Temperature parameterizations | High variety | Medium, exoplanet focused | Preserve concepts, redesign API. |
| Abundance/VMR parameterizations | High variety | Medium | Preserve concepts, redesign transforms. |
| Opacity table reading | High | Medium | Redesign I/O, port formats with fixtures. |
| Correlated-k random overlap | High | Medium/high | Port with strict reference tests. |
| Line-by-line support | High in NEMESIS | Minimal in NemesisPy | Preserve as later backend boundary. |
| CIA | High | Medium, partial TODOs | Port with reference tables and tests. |
| Rayleigh scattering | High | Medium | Port as separate opacity provider. |
| Clouds/aerosols | High but entangled | Medium/prototype | Preserve validated models, redesign. |
| Radiative transfer | High | Medium/high for exoplanet kernels | Port/modernize after benchmark suite. |
| Jacobians | High in NEMESIS | Limited | Redesign derivative strategy. |
| Retrieval likelihoods | High OE | Script-level Bayesian | Redesign around explicit likelihood objects. |
| Prior transforms | Legacy apriori/covariance | Config/script-level | Redesign. |
| Bayesian samplers | Historical add-ons/examples | Script-level MultiNest | Adapter architecture. |
| Instrument convolution | High legacy support | Medium k-table/bin support | Redesign I/O and response model. |
| Multi-instrument handling | Supported through multiple spectra/geometries | Emerging through observation lists | Redesign as datasets and likelihood blocks. |
| Stellar contamination / TLSE | Not central | Prototype modular engine | Preserve concept, isolate model. |
| GCM/phase-curve support | Some disc/geometric variants | Medium | Preserve concepts, redesign data model. |

## Atmospheric State Representation

What it does:

- Represents pressure, temperature, composition, aerosol/cloud, surface, radius/gravity, and derived layer/path quantities.

Why it exists:

- Radiative transfer needs profiles on a consistent vertical grid and retrieval needs a compact state vector mapped to physical profiles.

NEMESIS:

- Uses compact state vector `xn`, apriori `xa`, covariance `sa`, and `varident/varparam` integer-coded parameterizations.
- `subprofretg.f` maps retrieval vector elements into `.prf`, aerosol, cloud, surface, and other profile files.
- Scientifically mature but hard to reason about because meaning is carried by integer codes, common blocks, and side effects.

NemesisPy:

- Uses arrays for pressure, temperature, VMR, height, layer properties.
- Older backend stores state in mutable `ForwardModel` attributes.
- Modular layer builds state from raw config dictionaries.

ROBERT:

- Redesign as explicit typed data classes: `AtmosphereProfile`, `LayerGrid`, `Composition`, `CloudState`, `PlanetState`.
- Keep retrieval parameter vector separate from physical atmosphere.
- Parameterizations should be pure transforms from named parameters to physical state.

## Pressure Grids and Layering

What it does:

- Converts model-level atmospheric profiles into radiative-transfer layers and line-of-sight path amounts.

Why it exists:

- Radiative transfer operates on finite layers with representative pressure, temperature, absorber amount, and path scaling.

NEMESIS:

- `radtran/path/` handles layer construction, path geometry, tangent heights, spherical paths, and integration choices.
- Supports many observing geometries.

NemesisPy:

- `calc_layer.py` supports equal pressure, log pressure, height, path-length, and custom pressure/height layer bases.
- Layer averages use absorber-amount weighting and Simpson integration.

Maturity:

- NEMESIS is mature; NemesisPy is practical and readable but needs more validation coverage.

ROBERT:

- Port with tests.
- Make grid orientation explicit (`top_to_bottom` vs `bottom_to_top`).
- Require units at the boundary and test monotonicity/in-range behavior.

## Hydrostatic Height and Mean Molecular Weight

What it does:

- Computes altitude from pressure, temperature, gravity, radius, and mean molecular weight.

Why it exists:

- Emission and transmission path lengths depend on atmospheric scale height and radius reference.

NEMESIS:

- Has `XhydrostatH.f`, `XhydrostatP.f`, `newgrav.f`, and gas-giant reference radius handling.

NemesisPy:

- `common/calc_hydrostat.py` and variants compute hydrostatic profiles.
- `radtran/calc_mmw.py` maps gas IDs to molecular weights.

Maturity:

- Mature concept, but implementation details need unit and convention tests.

ROBERT:

- Port with tests against canonical profiles and NEMESIS/NemesisPy fixtures.
- Keep reference pressure/radius convention explicit.

## Temperature Parameterizations

What it does:

- Maps retrieval parameters into T(P) profiles.

Why it exists:

- Atmospheric thermal structure dominates emission spectra and degeneracies.

NEMESIS:

- `subprofretg.f` contains many model IDs.
- Includes Parmentier/Guillot-style helpers (`parmentierguillot1.f`), Robinson-Catling (`tbrownrc.f`), fixed/profile scaling variants, and many historical model codes.

NemesisPy:

- `models/TP_profiles.py` and `modular/engines/temperature.py` include isothermal, Guillot, Parmentier & Guillot 2014-like, and Madhusudhan-Seager style profiles.
- The modular implementation includes approximation notes, old commented versions, and debugger imports.

Maturity:

- Individual physical forms are well established, but the implementation interfaces are uneven.

ROBERT:

- Preserve scientifically used parameterizations.
- Redesign as independent `TemperatureParameterization` objects with named parameters, support bounds, and citation metadata.
- Treat experimental approximations as provisional.

## Abundance and Chemistry Parameterizations

What it does:

- Builds VMR(P) profiles from free abundances, background gases, quench prescriptions, equilibrium chemistry, or photochemical overlays.

Why it exists:

- Composition controls gas opacity and CIA; retrieval parameters usually describe composition compactly.

NEMESIS:

- Many gas/profile model IDs encoded in `subprofretg.f`.
- Supports log variables, scaling, continuous profiles, cloud/gas special cases, and previous-retrieval propagation.

NemesisPy:

- `models/gas_profiles.py` implements free VMR profile generation.
- `modular/engines/chemistry.py` supports free chemistry, FastChem interface, SO2/H2S photochemical profile, elemental quench, and species quench.

Maturity:

- NEMESIS is broad but opaque. NemesisPy is more exoplanet-relevant but prototype-like.

ROBERT:

- Redesign parameter transforms as composable objects.
- Separate chemistry engines from retrieval priors and from opacity gas ordering.
- Enforce VMR normalization/background-gas policies explicitly.

## Opacity Handling

What it does:

- Reads, stores, interpolates, and rebins gas opacity tables, line tables, cross sections, and CIA tables.

Why it exists:

- Opacity data dominate forward-model correctness and runtime.

NEMESIS:

- Mature Radtran opacity tooling in `radtran/radtran/`, `spec_data/`, `ciatable/`, and `raddata/`.
- Supports k-tables, LBL tables, continuum, CIA, partition functions, and line database tooling.

NemesisPy:

- `radtran/read.py` reads `.kta` and CIA files.
- `modular/io/ktables.py` uses `exo_k` for k-table fetching/rebinning and includes MPI-aware file handling.

Maturity:

- NEMESIS formats and routines are the reference.
- NemesisPy is useful but package dependencies and file policies need cleanup.

ROBERT:

- Redesign opacity I/O behind `OpacityDatabase` and `OpacityProvider` interfaces.
- Support NEMESIS `.kta`/CIA formats through tested readers.
- Keep rebinning/caching outside the radiative-transfer kernel.

## Correlated-k and Random Overlap

What it does:

- Combines gas k-distributions over g-ordinates to estimate absorption efficiently.

Why it exists:

- Full line-by-line calculations are expensive for retrieval loops.

NEMESIS:

- `get_kg.f`, `noverlapg.f`, `rankg.f`, `cirsradg_wave.f`.
- Scientifically mature and central.

NemesisPy:

- `calc_tau_gas.py` ports k interpolation and random-overlap logic with Numba.
- Notes indicate pressure/temperature out-of-range behavior and NEMESIS opacity scaling.

Maturity:

- High scientific importance; must be reference-tested.

ROBERT:

- Port with exacting tests against NEMESIS/NemesisPy fixtures.
- Make interpolation/extrapolation policy explicit.
- Benchmark memory layout and vectorization before changing algorithms.

## Line-by-Line

What it does:

- Computes opacity from spectral lines directly or via LBL tables.

Why it exists:

- Needed for validation, high-resolution work, and k-table generation.

NEMESIS:

- Strong support through `radtran/spec_data/`, `read_klbl*`, `lbl*`, and line-processing tools.

NemesisPy:

- Primarily correlated-k; LBL support is not central.

ROBERT:

- Preserve the backend boundary, but do not implement initially unless needed for validation.
- Design `OpacityProvider` so LBL can be added later without touching retrieval code.

## Collision-Induced Absorption

What it does:

- Adds absorption from molecular pairs such as H2-H2, H2-He, H2-CH4, N2-N2, etc.

Why it exists:

- CIA is essential in H2/He atmospheres and broad-band emission/transmission spectra.

NEMESIS:

- Mature CIA table generation and interpolation in `ciatable/`, `nciacon.f`, `nparacon_all.f`, and `raddata`.

NemesisPy:

- `calc_tau_cia.py` handles several pairs and partial H-minus opacity.
- Some comments mark CO2/N2 NIR CIA as TODO.

ROBERT:

- Port with table fixtures.
- Represent each CIA pair as a named opacity contributor with coverage metadata.
- Reject or warn on missing pair coverage explicitly.

## Rayleigh Scattering

What it does:

- Adds molecular scattering optical depth.

Why it exists:

- Important in visible/near-IR and transmission spectra.

NEMESIS:

- Several Rayleigh routines for different planetary atmospheres in `radtran/scatter/`.

NemesisPy:

- `calc_tau_rayleigh.py` provides simplified optical-depth contribution.

ROBERT:

- Port as an independent opacity/scattering provider.
- Keep scattering source-function treatment separate from extinction-only opacity.

## Clouds and Aerosols

What it does:

- Represents aerosol/cloud opacity, cross sections, phase functions, decks, haze power laws, patchiness, and size distributions.

Why it exists:

- Clouds and hazes strongly affect spectra and retrieval degeneracies.

NEMESIS:

- Broad mature support through aerosol files, cloud models, Mie/scattering utilities, `ackermanmarley*`, `mod_scatter`, `getcloud21`, `gsetrad`, and scattering kernels.
- Implementation is deeply tied to files and profile model IDs.

NemesisPy:

- `calc_tau_cloud.py` and `modular/engines/clouds.py` support analytic haze/Rayleigh and patchy clouds; species-informed cross sections use HDF5 and interpolation.

Maturity:

- NEMESIS scientific concepts are mature; NemesisPy cloud engine is useful but still application-shaped.

ROBERT:

- Redesign cloud APIs.
- Preserve physically validated NEMESIS cloud concepts but port incrementally with reference tests.
- Separate cloud vertical profiles, optical properties, and radiative-transfer scattering treatment.

## Radiative Transfer

What it does:

- Computes emergent radiance, flux, eclipse depth, transmission depth, contribution/weighting functions, and gradients.

Why it exists:

- This is the core forward model.

NEMESIS:

- `cirsradg_wave.f` is a mature gradient correlated-k engine.
- `cirsrad_wave.f` and scattering routines cover additional modes.

NemesisPy:

- `calc_radiance.py` computes emission, transmission, weighting functions, and cloud variants using Numba.
- Emission path integrates layer source terms and ground/deep contribution over g-ordinates.

Maturity:

- NEMESIS is the reference. NemesisPy is a useful exoplanet port but should be checked for every supported mode.

ROBERT:

- Initially implement a small subset only after docs are reviewed.
- Build golden reference tests before modernization.
- Keep radiative transfer free of sampler/I/O concerns.

## Jacobians and Weighting Functions

What it does:

- Computes derivatives of spectra with respect to state variables for optimal estimation and diagnostics.

Why it exists:

- Required for NEMESIS optimal estimation, covariance, averaging kernels, and contribution functions.

NEMESIS:

- Mature gradient pipeline maps layer derivatives to profile variables to state-vector variables via `map2pro` and `map2xvec`.
- Finite-difference fallback in `forwardnogX.f`.

NemesisPy:

- Has weighting functions, but not a full mature retrieval-Jacobian framework.

ROBERT:

- Redesign derivative strategy.
- Support finite differences first for validation, then consider analytic/JAX/AD/adjoint where scientifically safe.
- Keep derivative outputs as named structured arrays.

## Retrieval Likelihoods and Priors

What it does:

- Defines how model spectra are compared with data and how parameter space is constrained.

Why it exists:

- Retrieval correctness is as much statistical as radiative.

NEMESIS:

- Optimal-estimation cost combines measurement fit and apriori covariance using `calc_phiret.f`, `calc_gain_matrix.f`, `calc_serr.f`, and `calcnextxn.f`.
- Support constraints are embedded in `coreret.f`.

NemesisPy:

- Modular prior extraction exists in `modular/rt/setup.py`.
- MultiNest examples define priors and likelihoods inline in scripts.

ROBERT:

- Redesign.
- Provide explicit `Prior`, `Parameter`, `PriorTransform`, `Likelihood`, and `NoiseModel` abstractions.
- Treat optimal estimation and Bayesian sampling as two retrieval algorithms sharing the same forward/likelihood interfaces.

## Instrument Convolution and Multi-Instrument Handling

What it does:

- Maps model spectra to observed bins, instrument line-spread functions, multiple spectra, and offsets/jitter.

Why it exists:

- JWST observations are instrument/bin dependent and frequently multi-visit or multi-instrument.

NEMESIS:

- FWHM convolution, variable FWHM files, multiple geometries/spectra, and many observation formats.

NemesisPy:

- Observation loading, bin width inference, k-table rebinning, offsets, and multiple observations are emerging.

ROBERT:

- Redesign as `Dataset`, `SpectralOrder`, `InstrumentResponse`, and `CalibrationParameter` objects.
- Keep instrument convolution outside physics kernels.

## Stellar and System Models

What it does:

- Provides stellar flux, dilution, stellar contamination/TLSE corrections, planet/star geometry.

Why it exists:

- Exoplanet emission and transmission outputs are ratios to stellar flux and area.

NEMESIS:

- Solar/stellar reference files and secondary-eclipse flux ratio modes.

NemesisPy:

- Stellar blackbody/PHOENIX and TLSE machinery exists in modular I/O/engines.

ROBERT:

- Preserve concept.
- Redesign as separate stellar model and contamination model components with clear units.

