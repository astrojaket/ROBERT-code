# Taylor et al. (2021) cloud-scattering Figures 1–2 benchmark

`examples/benchmark_taylor2021_figures_1_2.py` recreates the controlled thermal-
scattering experiments shown in Figures 1 and 2 of Taylor et al. (2021), using
ROBERT's Toon thermal multiple-scattering solver.

Run it with:

```bash
python examples/benchmark_taylor2021_figures_1_2.py
```

The output directory contains ROBERT versions of both figures and a JSON file
recording the controlled inputs and the optically thick isothermal dilution
check.

The Figure 1 NEMESIS spectra and exact Figure 2 pressure-temperature profiles
are stored under `data/taylor2021_cloud_paper/` with provenance. ROBERT uses the
existing local petitRADTRANS R=1000 HDF5 H2O POKAZATEL and CO HITEMP
correlated-k tables together with H2-H2 and H2-He CIA. The benchmark therefore
compares independent opacity sources and RT implementations rather than
silently substituting a synthetic molecular template.

Figure 1 uses the archived isothermal 1400 K atmosphere, cloud
single-scattering albedo 0.9, and `log10(kappa_cld) = 0..8`. The output overlays
ROBERT and NEMESIS and reports residual metrics for every opacity case.
Because the NEMESIS `kappa_cld` parameter is not the same quantity as ROBERT's
bulk-atmosphere mass-extinction coefficient, the benchmark records an explicit
`-4 dex` mapping between their logarithmic opacity scales. This single offset
aligns the cloud/gas-opacity transition; it is a benchmark convention, not a
claim of unit equivalence.

Figure 2 fixes the grey cloud column optical depth and varies its SSA for a
non-inverted, isothermal, and inverted temperature profile. The benchmark is
expected to reproduce the qualitative result of the paper: thermal scattering
changes feature contrast differently depending on the sign and magnitude of
the temperature gradient.

The individual numerical Figure 2 spectra were not present in the downloaded
`Forward_models` bundle; only their PDFs were archived. ROBERT therefore uses
the exact archived TP profiles and cloud/SSA grid, while the retained PDFs are
the visual reference. A pointwise Figure 2 residual benchmark will require the
original individual NEMESIS spectral files.
