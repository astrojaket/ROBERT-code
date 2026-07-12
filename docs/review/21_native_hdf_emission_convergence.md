# Native HDF Emission and Pressure-Grid Convergence

## Purpose

The shared-opacity benchmarks establish the thermal RT solver independently of
opacity interpolation. This comparison closes the next boundary: ROBERT loads
all six pRT-format ExoMol correlated-k HDF files and both pRT-format CIA HDF
files itself, evaluates them on a genuinely cell-centred atmosphere, and
compares the resulting emission spectrum with stable petitRADTRANS 3.3.3.

No pRT-evaluated opacity or optical depth is passed to ROBERT in this test.

## Matched physical grids

ROBERT is a finite-volume model: pressure edges bound physical layers and
opacity is evaluated at layer centres. pRT3 treats its supplied pressure array
as radiative-transfer nodes. The paired grids therefore use the same physical
top and bottom boundaries but not falsely identical array semantics:

| ROBERT layers | pRT3 nodes |
|---:|---:|
| 40 | 41 |
| 80 | 81 |
| 160 | 161 |

The pressure boundaries span `1e-5` to 100 bar. ROBERT layer pressures are the
geometric means of adjacent edges. The analytic 900--1800 K temperature profile
is evaluated at centres and edges in ROBERT and at nodes in pRT3. Both codes use
15 m/s2 gravity, the same mass/volume-converted composition, and the exact
16-point pRT correlated-k quadrature.

## Native opacity path

ROBERT independently loads:

- H2O POKAZATEL;
- CO HITEMP;
- CO2 UCL-4000;
- CH4 YT34to10;
- NH3 CoYuTe;
- HCN Harris;
- H2-H2 CIA; and
- H2-He CIA.

Line coefficients are interpolated bilinearly in log(k), log-pressure, and
temperature. The new pRT CIA loader retains each collision pair as a separate
physical table. CIA alpha is interpolated linearly in log(alpha) with
temperature, as in pRT3, and linearly in wavenumber. The resulting CIA optical
depth uses the local ideal-gas density, hydrostatic path length, and explicit
H2/H2 or H2/He VMR product.

The HDF correlated-k coefficients remain on their exact stored wavenumber-bin
centres. pRT3 reports flux at frequency-bin averages, which are close but not
identical. Only the final pRT flux is interpolated onto the exact opacity-bin
centres; correlated-k coefficients are never spectrally interpolated.

## Paired results

| ROBERT/pRT grid | Median difference | RMS difference | Maximum difference |
|---|---:|---:|---:|
| 40 cells / 41 nodes | -0.160% | 0.295% | 0.947% |
| 80 cells / 81 nodes | +0.00066% | 0.108% | 0.647% |
| 160 cells / 161 nodes | +0.0335% | 0.110% | 0.594% |

The 80- and 160-layer comparisons have essentially the same approximately
0.11% RMS floor. This is consistent with the remaining difference between
ROBERT's analytic linear-source formal integration and pRT3's Feautrier source
and boundary discretization. It is not evidence of opacity non-convergence.

## Independent self-convergence

| Code | Coarse/fine grids | RMS spectral change | Maximum change |
|---|---|---:|---:|
| ROBERT | 40/80 cells | 0.163% | 0.327% |
| ROBERT | 80/160 cells | 0.0408% | 0.0817% |
| pRT3 | 41/81 nodes | 0.0956% | 0.372% |
| pRT3 | 81/161 nodes | 0.0232% | 0.0885% |

Both solvers reduce their RMS change by almost exactly a factor of four when
the pressure spacing is halved. This is regular second-order convergence. The
earlier comparison that treated pRT pressure nodes as ROBERT cell centres was
therefore a discretization mismatch, not a physical disagreement.

## Timing and memory boundary

Loading six large HDF line tables sequentially and evaluating all three ROBERT
grids takes 10.1 s. Sequential loading avoids retaining the roughly 2 GB source
table collection in memory simultaneously. Steady end-to-end ROBERT times,
including random overlap, native CIA, and RT, are:

| Layers | ROBERT steady time |
|---:|---:|
| 40 | 1.14 s |
| 80 | 2.29 s |
| 160 | 4.55 s |

The approximately linear resolution scaling is good. Stable pRT3 remains
faster, so repeated retrieval evaluation should cache evaluated opacity and
optimize the random-overlap reduction without changing its distribution
physics.

## Reproduction

Generate the small pRT convergence references:

```bash
/Users/jaketaylor/miniforge3/envs/petitradtrans-stable/bin/python \
  examples/run_petitradtrans3_multispecies_emission.py \
  opacity_data/petitRADTRANS/input_data \
  examples/outputs/petitradtrans3_stable/native_convergence_prt_nodes_41.npz \
  --layers 41 --flux-only
```

Repeat for 81 and 161 nodes, then run:

```bash
/Users/jaketaylor/miniforge3/envs/robert-exoplanets/bin/python \
  examples/benchmark_native_hdf_emission_convergence.py
```

The JSON report, spectra, and four-panel convergence plot are written below
`examples/outputs/petitradtrans3_stable/`.

## Conclusion

This establishes an end-to-end, clear-atmosphere emission path from downloaded
pRT/ExoMol HDF opacity through ROBERT-native interpolation, hydrostatic optical
depth, random overlap, CIA, and thermal RT. Within this six-molecule+CIA,
0.3--12 micron domain, both the emission solver and the native opacity path are
science-ready.

The next emission milestone is no longer another clear-spectrum RT comparison.
It is retrieval-scale performance work followed by an end-to-end cloudy
thermal-emission benchmark using native molecular opacity plus physical cloud
extinction, single-scattering albedo, phase moments, and the validated SH4
backend.
