# Model Setup

ROBERT now has a small model-setup factory that turns a configuration mapping
into atmosphere components:

- `PressureGrid`
- `TemperatureProfile`
- `ChemistryModel`
- optional `MeanMolecularWeightModel`
- default parameter values for a forward call

The public helper is:

```python
setup = build_atmosphere_setup(config, planet=planet)
builder = setup.build_atmosphere_builder()
atmosphere = builder.build(setup.default_parameters)
```

## Why This Exists

The core physics classes should not read raw dictionaries. Config dictionaries
belong at the I/O boundary. The setup factory lets us support practical model
configuration while keeping temperature, chemistry, opacity, and future
radiative-transfer code modular and testable.

## HAT-P-32b-Style Compatibility

The factory understands the model-setup structure used by the local HAT-P-32b
emission workflow:

```python
config = {
    "pressure_grid": {
        "n_layers": 100,
        "p_top_bar": 1.0e-6,
        "p_bot_bar": 100.0,
    },
    "temperature_profile": {
        "type": "guillot14",
        "values": {
            "kappa_IR": 0.02,
            "gamma1": 0.5,
            "gamma2": 1.5,
            "T_irr": 1500.0,
            "alpha": 0.5,
            "T_int": 200.0,
        },
    },
    "molecules": {
        "free": {
            "names": ["H2O", "CO"],
            "inactive": {"names": ["H2", "He"]},
            "log": True,
            "values": {"H2O": 1.0e-3, "CO": 1.0e-4},
        },
    },
}
```

In this compatibility layer, `values` are treated as physical default parameter
values for forward-model evaluation. They are not hard-coded into the physics
objects unless a component explicitly uses a fixed setting, such as
`T_int` in the current `guillot14` setup.

## Current Supported Types

Temperature:

- `isothermal`
- `tabulated`
- `madhu` / `madhusudhan_seager_2009`
- `guillot14` / `parmentier_guillot_2014`
- `spline`

Chemistry:

- free constant-with-altitude active gases
- explicit inactive/background gases, including H2/He default splitting

Mean molecular weight:

- composition-derived MMW for complete VMR budgets
- fixed MMW
- no MMW model, falling back to the builder's fixed scalar value

## Not Yet Included

The setup factory does not yet build:

- opacity providers from file paths
- instrument response objects
- radiative-transfer backends
- sampler/retrieval problem objects
- FastChem or quench chemistry adapters

Those should be added as separate factory functions once their underlying
domain objects are stable.
