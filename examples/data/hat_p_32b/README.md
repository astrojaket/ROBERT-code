# Bundled HAT-P-32b Retrieval Benchmark

This directory makes the validated ROBERT/NemesisPy comparison runnable from
an ordinary Git clone without personal paths or external opacity downloads.

Contents:

- `reference/`: saved NemesisPy HAT-P-32b observation and posterior products.
- `opacities/`: six ROBERT correlated-k archives prepared from the original
  ExoMolOP R1000 KTA files with `exo_k.bin_down_cp` on the 117 G395H bins.
- `fastchem/`: the FastChem input files required by this benchmark and the
  upstream GPL-3.0 license. The compiled `pyfastchem` runtime is installed by
  `environment.yml`, so the repository does not contain a platform-specific
  binary.
- `provenance.json` and `checksums.sha256`: generation metadata and integrity
  hashes.

The bundled opacity tables are specific to the reference G395H binning. They
do not replace ROBERT's generic ExoMol/exo_k ingestion path for new targets or
spectral grids. Maintainers with the native source products can regenerate the
bundle with `tools/build_hat_p_32b_bundle.py`.

FastChem is Copyright Daniel Kitzmann and Joachim Stock and is redistributed
under GPL-3.0. ROBERT's own source remains MIT licensed; third-party files keep
their stated licenses.
