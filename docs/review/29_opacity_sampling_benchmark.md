# ExoMol Opacity-Sampling Benchmark

Date: 2026-07-12

## Outcome

ROBERT now supports a beta, typed `OpacitySamplingProvider` alongside
`CorrelatedKOpacityProvider`. The new path reads the six requested ExoMolOP
TauREx cross-section grids, caches an explicit common wavelength subset during
model preparation, interpolates in log-pressure / temperature / log-cross
section, and combines gases by direct optical-depth addition. It does not call
the random-overlap kernel.

The speed hypothesis is confirmed. At equal native sample count (2,485 points,
1–12 micron, 40 layers, six gases), the warmed ROBERT calculation took 20.2 ms
with opacity sampling and 425 ms with correlated-k random overlap: a 21.0x
speedup. petitRADTRANS 3.3.3 took 14.6 ms and 158 ms respectively: a 10.9x
speedup.

The fastest setting is not the most accurate setting. After both spectra are
binned to R=100 and compared with a pRT3 R=15,000 ExoMolOP sampling reference:

| Method | Native effective R | RMS relative error | 95th percentile absolute error | Steady time |
|---|---:|---:|---:|---:|
| pRT3 opacity sampling, stride 15 | 1,000 | 4.88% | 9.23% | 14.6 ms |
| pRT3 opacity sampling, stride 5 | 3,000 | 2.42% | 4.64% | 43.0 ms |
| pRT3 opacity sampling, stride 3 | 5,000 | 1.73% | 3.47% | 72.0 ms |
| pRT3 correlated-k | 1,000, 16 g points | 1.60% | 3.18% | 158 ms |
| ROBERT opacity sampling, stride 3 | 5,000 | 1.77% | 3.69% | 102 ms |
| ROBERT correlated-k | 1,000, 16 g points | 1.58% | 3.10% | 425 ms |

The balanced setting in this benchmark is therefore sampling stride 3. It is
4.18x faster than ROBERT correlated-k while its R=100 RMS error is 1.77% versus
1.58% for correlated-k. Stride 15 is appropriate only when roughly 5% RMS
sampling error is acceptable.

ROBERT stride-15 sampling and pRT3 stride-15 sampling agree to 0.37% RMS after
R=100 binning. This is the cleanest cross-code check of the new loader and
direct-sum opacity path.

## Data and provenance

The six files are ExoMolOP cross sections at R=15,000 over 0.3–50 micron with
shape `(22 pressure, 27 temperature, 76,744 wavenumber)` and units
`cm^2/molecule`:

- H2O: POKAZATEL
- CO: Li2015
- CO2: UCL-4000
- CH4: YT34to10
- NH3: CoYuTe
- HCN: Harris

Exact byte sizes and SHA-256 checksums are recorded in
`docs/data/exomol_opacity_sampling_manifest.json`. The upstream CO2 HDF5 DOI
field contains the placeholder `qqq`; the manifest records the publication DOI
instead of silently correcting the source file.

The correlated-k comparison uses the already installed pRT/ExoMolOP tables.
Its CO table is HITEMP, not Li2015, so the pRT-to-pRT comparison does not isolate
only the numerical representation for that molecule. Five of six line-list
families match.

## Reproduction

```bash
python examples/download_exomol_opacity_sampling.py
python examples/setup_petitradtrans3_opacity_sampling_data.py

python examples/run_petitradtrans3_opacity_sampling_reference.py \
  opacity_data/exomol_pRT/input_data \
  examples/outputs/opacity_sampling/petitradtrans3_sampling3.npz \
  --sampling 3 --layers 40

NUMBA_CACHE_DIR=/tmp/robert-numba-cache \
python examples/benchmark_opacity_sampling.py
```

The pRT comparison requires the ExoMol files to be placed in pRT3's
`opacities/lines/line_by_line` hierarchy. POKAZATEL lacks the `mol_mass`
dataset required by pRT3.3.3; the local pRT comparison copy adds 18.01528 amu.
The downloaded source and its checksum are unchanged.

Machine: Apple arm64 (`T6000` kernel target), macOS/Darwin 24.6.0. Both ROBERT
and pRT environments used Python 3.12.13. Timings are medians of five warmed
calls; preparation and HDF5 loading are reported separately and excluded from
steady-call time.

## Interpretation

These results support classifying opacity sampling as a functional but
unfinished beta backend, not replacing correlated-k globally. Sampling accuracy depends on spectral
phase, output resolution, atmospheric state, abundances, and wavelength range.
Each science configuration should run a stride-convergence check against the
full cross-section grid before selecting its production stride.
