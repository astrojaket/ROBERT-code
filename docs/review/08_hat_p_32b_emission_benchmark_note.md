# HAT-P-32b Emission Benchmark Note

This is a local benchmark pointer for future ROBERT validation work. The data
are not part of this repository and should not be copied into version control
unless a tiny, licensed fixture is deliberately extracted later.

Local path on the current development laptop:

```text
/Users/jaketaylor/Dropbox/PostDoc4/Emission_Example/HAT-P-32b/
```

Observed contents on 2026-07-02:

- `emission/emission_R1000.csv`
- `emission/emission_index.csv`
- `emission/emission_R1000.png`
- `emission/pt_vmr.png`
- `kta_temp/H2O_emission_R1000.kta`
- `kta_temp/CO_emission_R1000.kta`
- `kta_temp/CO2_emission_R1000.kta`
- `kta_temp/CH4_emission_R1000.kta`
- `kta_temp/HCN_emission_R1000.kta`
- `kta_temp/NH3_emission_R1000.kta`

Likely future use:

- Treat the CSV emission spectrum as a comparison target once ROBERT has a
  validated opacity reader and reference emission solver.
- Treat the `.kta` files as external local inputs for exploratory development,
  not as committed test fixtures.
