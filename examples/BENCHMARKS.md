# Maintained Forward-Model Benchmarks

PICASO and petitRADTRANS are ROBERT's gold-standard independent forward-model
comparisons. Maintained benchmark plots use a shared purple visual language:
ROBERT and best-fitting model spectra use `mediumpurple`, related ROBERT curves
use darker or lighter purples, and external reference spectra use neutral dark
tones for contrast.

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

## NEMESIS

- `benchmark_jupiter_nemesis.py`: like-for-like reproduction of the official
  71-layer Jupiter CIRS nadir thermal-emission example. It reads NEMESIS's
  legacy KTA and CIA tables, explicit layer paths, gas columns, and Docker
  forward output without vendoring the large reference dataset.

The archived HAT-P-32b checks under `Depreciated_Benchmarks/` predate ROBERT's
YAML task workflow and are not maintained or run in CI.
