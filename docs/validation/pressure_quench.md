# Pressure-quench validation and oracle exchange

ROBERT's pressure-quench transform is unit- and integration-tested against
analytic profiles. That establishes the validation level **tested**. It does
not establish agreement with the research version of NemesisPy or reproduce a
published retrieval.

At implementation time, Taylor et al. (2026) defined the piecewise quench
profile but did not specify the off-grid numerical convention. The paper's
data-availability statement says generated data are available on request, and
no authoritative public chemistry-profile oracle was found. ROBERT therefore
records its interpolation choice as an assumption and does not fabricate a
parity fixture.

## Cross-framework oracle contract

An authoritative NemesisPy export can establish `cross-framework validated`
after review. Store one compact UTF-8 JSON file under `tests/fixtures/` with
this versioned structure:

```json
{
  "schema_version": 1,
  "source": {
    "framework": "NemesisPy",
    "repository_url": "REPLACE_WITH_AUTHORITATIVE_SOURCE_URL",
    "git_commit": "full-commit-hash",
    "export_script_sha256": "sha256",
    "created_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
    "paper": "Taylor et al. 2026, arXiv:2607.06491"
  },
  "conventions": {
    "pressure_unit": "bar",
    "pressure_orientation": "increasing",
    "quench_parameter_semantics": "log10(P_q/bar)",
    "off_grid_sampling": "document-the-source-rule",
    "closure_policy": "no_renormalization"
  },
  "case": {
    "pressure_bar": [0.001, 0.01, 0.1, 1.0],
    "temperature_K": [1000.0, 1100.0, 1200.0, 1300.0],
    "base_parameters": {"metallicity": 0.0, "CtoO": 0.55},
    "groups": [
      {"pressure_parameter": "log_Pq_CO2", "species": ["CO2"]}
    ],
    "quench_parameters": {"log_Pq_CO2": -1.5},
    "base_vmr": {"CO2": [0.0, 0.0, 0.0, 0.0]},
    "quenched_vmr": {"CO2": [0.0, 0.0, 0.0, 0.0]}
  }
}
```

The zeros above describe the schema only and must not be committed as a
scientific oracle. An accepted fixture must contain the source's actual
full-precision values, a real immutable commit identifier, the exact export
script checksum, and an explicit off-grid rule. Keep the case small and
human-readable in accordance with `docs/data_policy.md`; do not commit opacity
tables, chains, or generated spectra.

The corresponding ROBERT test must:

1. reconstruct the pressure grid without reordering the fixture;
2. evaluate the same base chemistry inputs or load the exported base profiles;
3. compare both base and quenched profiles with documented tolerances;
4. test an exact-grid and an off-grid quench pressure; and
5. label the result `cross-framework validated` only if the numerical
   convention is equivalent or the expected difference is explicitly scoped.

A later successful end-to-end reproduction of a Taylor et al. retrieval is a
separate `science demonstrated` milestone and must record observation,
opacity, sampler, and posterior provenance in addition to this profile oracle.
