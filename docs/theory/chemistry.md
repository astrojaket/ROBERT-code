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
"He": 0.1453})`, matching the H2/He split used by the local NemesisPy
HAT-P-32b free-chemistry path. Background fractions are relative shares of the
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

## Not Yet Implemented

The current chemistry layer intentionally does not implement:

- FastChem or other equilibrium chemistry backends.
- Quench chemistry.
- Photochemical profile overlays.

Those should be added as separate adapters or post-processing components, not
as hidden branches inside the free-chemistry model.
