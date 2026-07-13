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
required FastChem inputs under `data/chemistry/fastchem`. Together
they pass a real smoke test, including repeated calls separated by a different
metallicity and C/O state. The FastChem path, parameter names, species mapping,
and elemental-abundance source are recorded in model provenance. FastChem
remains an optional dependency, each MPI process owns its own solver instance,
and the bundled upstream files retain their GPL-3.0 license.

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

The default normalization policy is strict: `normalization="require"` checks
that the VMRs sum to one in every layer. This is deliberate. A trace-only
composition should not silently define the bulk atmosphere. For controlled
comparison workflows, `normalization="normalize"` can renormalize the supplied
composition before calculating mean molecular weight.

`normalization="raw_sum"` is available for parity with the existing HAT-P-32b
FastChem workflow, where a selected set of returned species is used directly.

## Not Yet Implemented

The current chemistry layer intentionally does not implement:

- Quench chemistry.
- Photochemical profile overlays.
- Disequilibrium chemistry beyond tabulated or externally evaluated inputs.

Those should be added as separate adapters or post-processing components, not
as hidden branches inside the free-chemistry model.
