# Stable petitRADTRANS 3 Multispecies Emission Benchmark

## Scope

This benchmark compares ROBERT with stable petitRADTRANS 3.3.3 for clear
thermal emission from 0.3 to 12 micron. It uses the same atmosphere and opacity
set as the final transmission benchmark:

- 80 pressure points from `1e-5` to 100 bar;
- temperatures from 900 to 1800 K;
- constant gravity of 15 m/s2;
- 3689 wavelength bins and the exact 16-point pRT `petit_samples` quadrature;
- H2O, CO, CO2, CH4, NH3, and HCN line opacity; and
- H2-H2 and H2-He collision-induced absorption.

petitRADTRANS receives mass mixing ratios. ROBERT receives the exactly
converted volume mixing ratios, giving a mean molar mass of
`2.3242008615381975` amu. Stable pRT3 is the only petitRADTRANS reference; pRT4
is excluded.

Rayleigh scattering is deliberately absent from this thermal-emission case.
Rayleigh is scattering rather than true absorption, so adding it as absorbing
extinction would violate the thermal source equation. Its effect is negligible
where this 900--1800 K atmosphere emits appreciable flux, and any future
inclusion must use a scattering-capable thermal solver.

## Isolation hierarchy

The comparison separates two numerical boundaries.

### Shared vertical optical depth

pRT3 evaluates each species opacity and integrates it onto its pressure-node
grid. ROBERT receives the resulting species-resolved layer optical depths,
performs its own random-overlap reduction, and solves its linear-source formal
integral over an eight-angle Gaussian disc quadrature. This is the primary
test of correlated-k gas combination and thermal RT.

### Shared evaluated opacity

pRT3's returned, mass-weighted species opacities are converted back to
molecular cross-sections. ROBERT then constructs hydrostatic columns and
vertical optical depths itself. This additionally crosses a grid-convention
boundary: pRT treats the supplied pressures as RT nodes, while the ROBERT
`PressureGrid` treats them as finite-volume layer centers with derived edges.
The calculation is retained as an interface diagnostic, not presented as an
identical-discretization RT comparison.

## Flux results

| Band | Shared-tau RMS | Shared-tau maximum | Evaluated-opacity RMS | Evaluated-opacity maximum |
|---|---:|---:|---:|---:|
| 0.3--12 micron | 0.140% | 0.735% | 1.980% | 4.313% |
| 0.3--1 micron | 0.0919% | 0.204% | 2.174% | 4.313% |
| 1--5 micron | 0.165% | 0.735% | 2.222% | 3.326% |
| 5--12 micron | 0.145% | 0.611% | 0.972% | 1.674% |

The shared-tau spectrum validates the emission RT core for this chemically
complete clear-atmosphere case. Its median full-range offset is `-0.0869%`.
The residual is sub-percent everywhere and has the same scale expected from
ROBERT's analytic linear-in-optical-depth source integration versus the pRT3
Feautrier discretization and lower-boundary convention.

The evaluated-opacity difference is smooth, predominantly negative, and much
larger than the shared-tau result. Replacing only the vertical optical-depth
construction reduces the RMS from 1.980% to 0.140%; therefore it is not a gas
random-overlap or thermal-source discrepancy. A perfect-g-correlation
diagnostic does not improve the RMS and reaches an 8.16% maximum, independently
confirming that perfect correlation is not an acceptable substitute for
random overlap.

## Contribution functions

The pRT3 and ROBERT normalized contribution maps recover the same molecular
photospheric structure across the full infrared. Over 1--12 micron:

- the median peak-pressure displacement is `0.08861 dex`;
- the RMS peak-pressure displacement is `0.08251 dex`; and
- the centroid-pressure RMS displacement is `0.08557 dex`.

The pressure grid spacing is `7 / 79 = 0.08861 dex`, so the median displacement
is exactly one pRT pressure node. This is consistent with the node-versus-layer
convention rather than a displaced physical photosphere. No empirical pressure
shift is applied.

## Timing

On the Apple-silicon CPU:

| Calculation | First call | Steady median |
|---|---:|---:|
| pRT3 emission | 0.506 s with diagnostics | 0.486 s |
| ROBERT emission | 4.82 s | 2.13 s |

pRT3 includes its opacity evaluation and RT. ROBERT starts from pRT3-evaluated
species opacity but performs hydrostatic column construction, random-overlap
reduction, and RT. ROBERT is approximately 4.4 times slower at steady state.
Profiling should target the six-species random-overlap reduction while keeping
the validated distribution physics unchanged.

## Reproduction

```bash
/Users/jaketaylor/miniforge3/envs/petitradtrans-stable/bin/python \
  examples/run_petitradtrans3_multispecies_emission.py \
  opacity_data/petitRADTRANS/input_data \
  examples/outputs/petitradtrans3_stable/multispecies_emission_reference.npz

/Users/jaketaylor/miniforge3/envs/robert-exoplanets/bin/python \
  examples/benchmark_petitradtrans3_multispecies_emission.py
```

The scripts write the reference arrays, JSON metrics, compressed comparison
arrays, and a four-panel flux/contribution-function plot below
`examples/outputs/petitradtrans3_stable/`.

## Scientific conclusion

ROBERT's clear, absorption-dominated thermal-emission RT core is validated for
six molecular absorbers plus H2-H2/H2-He CIA over 0.3--12 micron. This result
does not yet validate independent end-to-end interpolation of every pRT HDF
table, chemical-equilibrium abundance generation, cloud opacity, or cloudy
multiple-scattering emission. Those are forward-model validation boundaries,
not failures of the thermal formal solver established here.

The next emission task should be a ROBERT-native opacity run on an explicitly
cell-centred pressure grid, paired with a pRT pressure-node convergence series.
That will validate the HDF interpolation and demonstrate convergence without
pretending that pressure nodes and finite-volume layer centers are identical.
