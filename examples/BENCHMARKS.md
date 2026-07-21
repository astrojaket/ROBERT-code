# Maintained Forward-Model Benchmarks

PICASO and petitRADTRANS are ROBERT's gold-standard independent forward-model
comparisons. Maintained benchmark plots use a shared purple visual language:
ROBERT and best-fitting model spectra use `mediumpurple`, related ROBERT curves
use darker or lighter purples, and external reference spectra use neutral dark
tones for contrast.

## Stellar spectra

- `benchmark_g_star_stellar_spectrum.py`: flux-conserving STScI PHOENIX
  profile for a Sun-like G2V star versus the explicit blackbody fallback,
  including the resulting secondary-eclipse normalization difference.
- `stellar_contamination_transmission.py`: runnable synthetic spot+facula TSLE
  example, using a controlled blackbody approximation by default or external
  STScI PHOENIX data when requested.
- `benchmark_poseidon_stellar_contamination.py`: commit-verified execution of
  the official POSEIDON v1.4 TSLE function bodies for homogeneous, spot,
  facula, and mixed cases. This isolates the transform and does not validate a
  stellar atmosphere grid. Run it with an official checkout outside ROBERT:

  ```bash
  python examples/benchmark_poseidon_stellar_contamination.py \
    --poseidon-source /path/to/POSEIDON
  ```

  Products are written beneath ignored `examples/outputs/`; the compact oracle
  and validation boundary are documented in
  `docs/theory/stellar_contamination.md`.

## PICASO

- `compare_grey_cloud_rt_picaso.py`: controlled grey-cloud radiative transfer.
- `benchmark_sh4_rt.py`: Toon and SH4 scattering comparisons.
- `benchmark_end_to_end_cloud_parity.py`: independent cloud-property and RT
  assembly.
- `benchmark_official_picaso_molecular_cloud_parity.py`: official
  molecular-opacity and cloudy-emission parity.

## petitRADTRANS

- `benchmark_petitradtrans3_stable.py`: stable emission and transmission
  comparison.
- `benchmark_petitradtrans3_multispecies_emission.py`: multi-species emission.
- `benchmark_petitradtrans3_multispecies_transmission.py`: multi-species
  transmission.
- `benchmark_petitradtrans4.py`: petitRADTRANS 4 comparison.
- `benchmark_native_hdf_emission_convergence.py`: native-grid emission
  convergence.

Superseded pre-YAML HAT-P-32b checks and their generated products have been
removed from Git. Released benchmark products belong in the Zenodo archive
described by `docs/data_policy.md`.
