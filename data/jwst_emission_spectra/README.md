# Published JWST exoplanet emission spectra

This directory is ROBERT's local index of refereed JWST thermal-emission
measurements for transiting exoplanets. The scope includes secondary-eclipse
spectra, phase-resolved emission spectra, derived nightside spectra, and
single-band eclipse photometry. Directly imaged planets and transmission-only
measurements are outside this catalog.

`catalog.yaml` is a paper-level index. It was assembled on 2026-07-13 from a
NASA Exoplanet Archive TAP query for JWST `Eclipse` products and a literature
cross-check for published products not yet represented there. `archive_product`
means the archive serves one or more reduced spectrum tables; `literature_only`
means the paper or its own repository is the current source.

Refresh the machine-readable archive snapshot with:

```bash
python scripts/sync_jwst_emission_catalog.py
```

Add `--download-spectra` to download every reduced table currently exposed by
the archive into `spectra/`. The sync writes SHA-256 checksums and never treats
different pipelines or visits as independent publications. Use the paper-level
catalog to choose a fiducial reduction deliberately before retrieval.

The NASA table changes over time, so "all" here means all records found within
the stated scope and search date, not a permanent completeness claim. Primary
resources are the [Atmospheric Spectroscopy Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/atmospheres/nph-firefly?atmospheres),
its [column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/atmospheres/atmospheres_columns.html),
and the cited paper/data links in `catalog.yaml`.
