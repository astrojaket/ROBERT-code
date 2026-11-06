# Independent FastChem elemental abundances

FastChem keeps its existing metallicity and C/O mode by default. To retrieve
individual elemental number ratios, set both legacy parameter names to `null`
and map each element to one parameter. Do not set either legacy parameter on
its own.

```yaml
atmosphere:
  chemistry:
    model: fastchem_equilibrium
    fastchem_path: data/chemistry/fastchem
    species:
      - {label: H2O, fastchem_name: H2O1}
      - {label: CO, fastchem_name: C1O1}
      - {label: SiO, fastchem_name: O1Si1}
      - {label: H2, fastchem_name: H2}
      - {label: He, fastchem_name: He}
    metallicity_parameter: null
    carbon_to_oxygen_parameter: null
    element_log10_number_ratio_parameters:
      O: log10_O_H
      C: log10_C_H
      Si: log10_Si_H

parameters:
  - name: log10_O_H
    unit: dex
    prior: {type: uniform, lower: -3.8, upper: -3.0}
  - name: log10_C_H
    unit: dex
    prior: {type: uniform, lower: -4.2, upper: -3.0}
  - name: log10_Si_H
    unit: dex
    prior: {type: uniform, lower: -5.0, upper: -4.0}
```

Each parameter is `log10(n_element / n_H)`. For example,
`log10_O_H: -3` sets `n_O / n_H` to `10^-3`. The model starts from the solar
abundance vector for every evaluation. It changes only the mapped elements.
Hydrogen, helium, and all unlisted elements keep their solar values. Independent
mode does not apply a global metallicity scale or set carbon from C/O.

The configured FastChem abundance file must contain each mapped element. The
configuration rejects unknown element symbols and H/He overrides. Existing
constant species overrides and pressure quenching continue to apply. The
resulting VMR profiles come from FastChem's equilibrium calculation; this
configuration change does not by itself establish physical validation for a
retrieval.

The [6 November local recovery record](../review/59_local_elemental_recovery_2026-11-06.md)
reports matched 50- and 100-live-point fixed-temperature injections on four
CPUs, run to convergence without iteration caps. All three truths lie inside
their 1-sigma and 2-sigma intervals, but C/H and Si/H remain broad. This is
one controlled noise realization, not target-science or calibration evidence.
The configured H-minus contribution is zero in that case.
