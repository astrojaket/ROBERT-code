# Taylor et al. (2021) cloud-paper benchmark data

These files were copied from Jake Taylor's Dropbox archive for *How does
thermal scattering shape the infrared spectra of cloudy exoplanets?* (MNRAS
506, 1309; DOI `10.1093/mnras/stab1854`).

- `figure1/wasp121_*.mre`: the nine NEMESIS forward spectra used by the
  archived `plot_diff_opac.py`, corresponding to `log10(kappa_cld) = 0..8`.
  The historical filename says `wasp121`; the system parameters used in the
  paper setup are the WASP-43-like values recorded by the benchmark script.
- `figure2/wasp121.txt`: isothermal atmosphere.
- `figure2/wasp121_profile.txt`: non-inverted atmosphere.
- `figure2/wasp121_profile_inv.txt`: inverted atmosphere.
- `figure2/compare_*_3.pdf`: archived numeric plots for the three Figure 2
  spectral panels. The underlying individual `.mre` spectra were not present
  in the downloaded `Forward_models` bundle, so these PDFs are retained as the
  visual references.

ROBERT does not redistribute the 1.9 GB opacity installation. The benchmark
loads the existing local petitRADTRANS R=1000 HDF5 H2O/CO and CIA tables and
records their resolved paths in its JSON report.
