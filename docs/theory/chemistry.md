# Chemistry

ROBERT treats chemistry models as modular components that evaluate composition
on a `PressureGrid` after a temperature profile has been evaluated. The
radiative-transfer layer should receive an atmospheric state and should not need
to know whether the composition came from free chemistry, an equilibrium
adapter, or a future disequilibrium model.

Current chemistry types:

- `ConstantChemistry`: repeats fixed mixing ratios through every atmospheric
  layer. This remains useful for wiring tests and compact examples.
- `BackgroundGasMixture`: describes how unused volume-mixing-ratio budget is
  split between inactive/background gases.
- `FreeChemistry`: evaluates layer-constant free abundances for active gases
  and can fill the remaining VMR budget with a background mixture.
- `CompositionMeanMolecularWeight`: derives layer mean molecular weight from
  evaluated VMR profiles and molecular masses.
- `FixedMeanMolecularWeight`: returns one fixed mean molecular weight for every
  atmospheric layer.
- `FastChemEquilibriumChemistry`: optional adapter for local FastChem
  equilibrium-chemistry tables when the external dependency and data files are
  available.

## FastChem Retrieval Chemistry

The FastChem adapter is connected to the parameterized cloud-free retrieval
model. It accepts `[M/H]` in dex through `metallicity` and a linear elemental
carbon-to-oxygen ratio through `CtoO`. Every evaluation resets the solver from
stored solar elemental abundances before applying these parameters, so calls do
not inherit elemental abundances from the previous sample.

The Conda environment installs `pyfastchem`, and the repository bundles the
required lightweight FastChem inputs under `data/chemistry/fastchem`. Together
they pass a real smoke test, including repeated calls separated by a different
metallicity and C/O state. The FastChem path, parameter names, species mapping,
and elemental-abundance source are recorded in model provenance. FastChem
remains an optional dependency, each MPI process owns its own solver instance,
and the bundled upstream files retain their GPL-3.0 license.

## Pressure-quench chemistry

`PressureQuenchChemistry` is an immutable decorator around a base
`ChemistryModel`. It implements the parameterised transport-quench model in
[Taylor et al. (2026)](https://arxiv.org/abs/2607.06491): for each configured
group and quench pressure `P_q`, the base profile is retained at `P > P_q` and
is fixed to `X_eq(P_q)` at `P <= P_q`. The decorator does not contain FastChem
logic and radiative transfer sees only the resulting `AtmosphereState`.

Every group explicitly names one pressure parameter and one or more species:

- a one-species group is molecular quenching;
- a multi-species group applies one shared pressure to coupled/grouped
  molecular profiles; and
- a species may occur in only one group.

The Python API is species-agnostic. The selected names must be produced by the
base chemistry model, and the suitability of a molecule and prior depends on
the temperature regime and chemical pathways. CH4 and NH3 are examples from
the paper, not a whitelist. Strict YAML support initially decorates
`FastChemEquilibriumChemistry`; applying this transform to ROBERT's
constant-with-altitude free chemistry would be a numerical no-op and is not a
configured or validated science mode.

Pressure parameters always mean `log10(P_q / bar)`. Their priors are supplied
by the user. A uniform prior on this logarithmic parameter is not a
`log_uniform` prior on an already logarithmic value. ROBERT converts the
`PressureGrid` unit to bar explicitly and supports either grid orientation.
The allowed domain is the inclusive minimum-to-maximum range of layer-center
pressures. Values outside it raise an error; no extrapolation is performed.

Taylor et al. define the piecewise profile but do not state how an off-grid
`P_q` is sampled, and no authoritative NemesisPy oracle was available for this
implementation. ROBERT therefore makes an explicit assumption: interpolate
VMR linearly as a function of `log10(P / bar)` between layer centers. This
preserves non-negative and zero abundances without inventing a floor. It is a
ROBERT convention, not a claim of exact NemesisPy parity, and is recorded in
model and retrieval-manifest metadata.

Quenched profiles are never renormalized because that would change the
published parameterisation. `evaluate_with_diagnostics()` reports immutable
base and quenched per-layer VMR sums, their drift, and elemental-count drift
when every species label is a parseable molecular formula. If any label cannot
be parsed, elemental drift is omitted rather than guessed. The normal
`AtmosphereBuilder` path computes mean molecular weight after quenching.

The Taylor et al. hot-Jupiter grouped preset is:

- carbon: H2O, CO, CO2, and CH4 using `log_Pq_C`; and
- nitrogen: NH3 using `log_Pq_N`.

This is "elemental" only in the paper's sense of element-grouped molecular
profiles. It is not an elemental-conservation solver. N2 motivated the nitrogen
chemistry but was absent from the paper's opacity/species set, so the preset
does not silently add it. A custom Python group may include N2 when the base
model produces it.

Only the pressure-quench transform is implemented here. The H2S deep-abundance,
break-pressure, and power-law photochemical-depletion profile in Taylor et al.
is a separate parameterisation and is intentionally out of scope.

Unit and integration tests establish the validation level `tested`. No frozen
NemesisPy profile comparison is currently available, so the implementation is
not `cross-framework validated`; no retrieval reproduction has been run, so it
is not `science demonstrated`. The reproducible oracle exchange contract is
documented in [Pressure-quench validation](../validation/pressure_quench.md).

## Free Chemistry

`FreeChemistry` is the first retrieval-facing chemistry model in ROBERT. It is
designed to match the common exoplanet emission-retrieval pattern used in the
HAT-P-32b workflow: retrieve or fix active gas abundances, then assign the
remaining atmospheric budget to H2/He.

Active gas values can be supplied in two modes:

- `parameter_mode="linear"`: parameter values are interpreted directly as VMRs.
- `parameter_mode="log10"`: parameter values are interpreted as `log10(VMR)`.

Each active species can either be retrieved or fixed. Retrieved species declare
their parameter names through `required_parameters()`. Fixed species are stored
in `fixed_mixing_ratios` and do not appear in `required_parameters()`.

By default, `FreeChemistry` uses `BackgroundGasMixture({"H2": 0.8547,
"He": 0.1453})`, matching the H2/He split used by the local HAT-P-32b
free-chemistry benchmark path. Background fractions are relative shares of the
leftover VMR budget and are normalized during validation.

The default `excess_policy="raise"` rejects active-gas abundances that sum to
more than one. `excess_policy="normalize"` exists for controlled comparison
workflows, but it changes the meaning of the active abundance parameters and
should not be used silently in retrieval production runs.

### Centered-log-ratio priors

For nested-sampling free-chemistry retrievals, ROBERT supports joint
centered-log-ratio (CLR) priors with the same transform used by POSEIDON. With
`N` retrieved gases, the remaining background abundance is the `(N + 1)`th
composition category. The transform samples `N` independent CLR coordinates,
constructs the omitted coordinate from the zero-sum constraint, and maps the
result to a strictly positive, unit-sum composition. A lower limit of `-12`
therefore means every category, including the derived background, must have
`VMR >= 10^-12`.

All retrieved abundance parameters in a CLR configuration must use
`type: centered_log_ratio`, identical bounds, and the same `group`. Free
chemistry must use `parameter_mode: log10` and `fill_background: true`. This
prior is intrinsically joint, so it is supported by direct nested sampling and
is intentionally rejected for optimal estimation and hybrid retrievals.

```yaml
atmosphere:
  chemistry:
    model: free
    species: [SO2, CO2, H2S]
    parameter_mode: log10
    parameter_names: {SO2: log_SO2, CO2: log_CO2, H2S: log_H2S}
    background_species: [H2]
    background_fractions: [1.0]
    fill_background: true
parameters:
  - {name: log_SO2, prior: {type: centered_log_ratio, lower: -12, upper: 0, group: composition}}
  - {name: log_CO2, prior: {type: centered_log_ratio, lower: -12, upper: 0, group: composition}}
  - {name: log_H2S, prior: {type: centered_log_ratio, lower: -12, upper: 0, group: composition}}
```

The composition and MMW implementation is shared by the transmission and
emission forward models. The CLR path is first being scientifically validated
for transmission; its availability in emission should not be read as an
emission validation claim.

### Phantom background gas

A phantom (or ghost) molecule can replace an assumed physical background gas.
It fills the VMR left by retrieved gases, contributes no line or continuum
opacity of its own, and has a retrieved molecular mass. Its contribution to
the atmospheric mean molecular weight is therefore
`VMR_phantom * phantom_mmw`. This separates scale-height information from an
assumption such as an H2/He or N2 background.

Configure the phantom as the sole background species and provide its mass
parameter:

```yaml
atmosphere:
  chemistry:
    model: free
    species: [CO2, CH4]
    parameter_mode: log10
    parameter_names: {CO2: log_CO2, CH4: log_CH4}
    background_species: [phantom]
    background_fractions: [1.0]
    fill_background: true
    phantom_species: phantom
    phantom_mean_molecular_weight_parameter: phantom_mmw
parameters:
  - {name: phantom_mmw, unit: amu, prior: {type: uniform, lower: 2.3, upper: 100.0}}
```

When combined with CLR priors, the phantom is the derived final CLR category.
It is deliberately absent from opacity species, Rayleigh scattering, and CIA
pairs; only its fitted mass affects MMW and hence hydrostatic structure.

## Mean Molecular Weight

ROBERT can now derive mean molecular weight from the evaluated composition:

```python
builder = AtmosphereBuilder(
    pressure_grid=pressure_grid,
    temperature_profile=temperature_profile,
    chemistry_model=free_chemistry,
    mean_molecular_weight_model=CompositionMeanMolecularWeight(),
)
```

For VMR composition, the layer mean molecular weight is the VMR-weighted sum of
species molecular masses. The built-in mass table covers the common gases used
in the current free-chemistry workflow, including H2, He, H2O, CO, CO2, CH4,
NH3, HCN, SO2, and H2S. Additional or renamed species should be supplied through
the `molecular_masses` mapping.

For a phantom species, `molecular_mass_parameters` maps the species name to a
retrieval parameter. Static and retrieved species masses may coexist in the
same `CompositionMeanMolecularWeight` model.

The default normalization policy is strict: `normalization="require"` checks
that the VMRs sum to one in every layer. This is deliberate. A trace-only
composition should not silently define the bulk atmosphere. For controlled
comparison workflows, `normalization="normalize"` can renormalize the supplied
composition before calculating mean molecular weight.

`normalization="raw_sum"` is available for parity with the existing HAT-P-32b
FastChem workflow, where a selected set of returned species is used directly.

## Not Yet Implemented

The current chemistry layer intentionally does not implement:

- photochemical profile overlays, including the H2S break-pressure model; or
- kinetics-derived quench pressures or eddy-diffusion retrievals.

Those should be added as separate adapters or post-processing components, not
as hidden branches inside the free-chemistry model.
