# Maintained configurations

The maintained YAML files are grouped by use. Quick checks and validation
cases are in `quickstart/`, reusable examples and the annotated template are
in `examples/`, and canonical WASP target configurations are in `targets/`.
The target files use external R=1000 tables unless their name says R100. Do
not mix resolutions between checkpoints or prepared-opacity caches. R=15000
tables are not distributed with ROBERT; contact Jake Taylor directly for the
validated high-resolution data workflow.

The quickstart files are:

| Purpose | YAML |
| --- | --- |
| Bundled-opacity WASP-80b first forward model | `quickstart/wasp80b_cloud_free_native_pg14_R100.yaml` |
| R=100 emission injection recovery | `quickstart/r100_emission_validation.yaml` |
| R=100 transmission injection recovery | `quickstart/r100_transmission_validation.yaml` |

The two R=100 validation cases are used by
`examples/notebooks/r100_emission_transmission_validation.ipynb`. They use
MultiNest, 60 ppm uncertainties, 18 bins from 1.10–1.70 microns, and no
external opacity or stellar files.

The reusable examples and template are:

| Purpose | YAML |
| --- | --- |
| Complete annotated schema template | `examples/TEMPLATE_all_supported_options.yaml` |
| Official-PICASO JWST-like transmission benchmark | `examples/picaso_jwst_transmission_retrieval_multinest.yaml` |
| Single-molecule transmission recovery | `examples/synthetic_transmission_injection_recovery_multinest.yaml` |
| Six-molecule clear transmission recovery | `examples/synthetic_six_molecule_transmission_injection_recovery_multinest.yaml` |
| Six-molecule cloudy transmission recovery | `examples/synthetic_six_molecule_cloudy_transmission_injection_recovery_multinest.yaml` |

The target defaults use PHOENIX stellar-atmosphere spectra selected by
`effective_temperature_k`, `log_g_cgs`, and `metallicity_dex`. Set
`PYSYN_CDBS` to the STScI Synphot reference-data root before building a forward
model. Change `bodies.star.spectrum_model` to `blackbody` only for an explicit
Planck-spectrum comparison.

The WASP-69 configurations use the PHOENIX grid point at 4750 K,
`log(g)=4.5`, and `[M/H]=0.0`. Their active gas opacity set is H2O, CO2, CO,
CH4, NH3, and SO2; SO2 is a retrieved constant-abundance override with the
same prior in every sampler variant.

| Target | Scenario | Default YAML |
| --- | --- | --- |
| WASP-69b | Cloud-free, native F322W2/F444W/LRS, PG14 | `targets/WASP-69b/wasp69b_cloud_free_native_pg14_R1000.yaml` |
| WASP-69b | Cloud-free, NIRCam F322W2/F444W, PG14 | `targets/WASP-69b/wasp69b_cloud_free_nircam_pg14_R1000.yaml` |
| WASP-69b | Cloud-free, native modes, retrieved isothermal T-P | `targets/WASP-69b/wasp69b_cloud_free_native_isothermal_R1000.yaml` |
| WASP-69b | Fixed MgSiO3 catalogue Mie cloud, PG14 | `targets/WASP-69b/wasp69b_mie_catalog_pg14_R1000.yaml` |
| WASP-69b | Retrieved Mie n/k cloud, PG14 | `targets/WASP-69b/wasp69b_mie_direct_nk_pg14_R1000.yaml` |
| WASP-80b | Cloud-free, native F322W2/F444W/LRS, PG14 | `targets/WASP-80b/wasp80b_cloud_free_native_pg14_R1000.yaml` |
| WASP-80b | Cloud-free, NIRCam F322W2/F444W, PG14 | `targets/WASP-80b/wasp80b_cloud_free_nircam_pg14_R1000.yaml` |
| WASP-80b | Cloud-free, native modes, retrieved isothermal T-P | `targets/WASP-80b/wasp80b_cloud_free_native_isothermal_R1000.yaml` |
| WASP-80b | Fixed MgSiO3 catalogue Mie cloud, PG14 | `targets/WASP-80b/wasp80b_mie_catalog_pg14_R1000.yaml` |
| WASP-80b | Retrieved Mie n/k cloud, PG14 | `targets/WASP-80b/wasp80b_mie_direct_nk_pg14_R1000.yaml` |

The WASP target directories also contain the matched regional-emission
MultiNest matrix. Each matrix file extends the corresponding target science
configuration, so only the matrix run identity and controls change. Writable
directories are local defaults beside each matrix `configuration.yaml`:

| Workflow | Cloud-free YAML | Mie-catalogue YAML |
| --- | --- | --- |
| Canonical MultiNest science | `targets/WASP-69b/wasp69b_cloud_free_native_pg14_R1000.yaml` | `targets/WASP-69b/wasp69b_mie_catalog_pg14_R1000.yaml` |
| Optimal estimation | `targets/WASP-69b/wasp69b_cloud_free_native_pg14_R1000_optimal_estimation.yaml` | `targets/WASP-69b/wasp69b_mie_catalog_pg14_R1000_optimal_estimation.yaml` |
| OE to MultiNest | `targets/WASP-69b/wasp69b_cloud_free_native_pg14_R1000_optimal_estimation_to_multinest.yaml` | `targets/WASP-69b/wasp69b_mie_catalog_pg14_R1000_optimal_estimation_to_multinest.yaml` |

The Mie-catalogue case also provides
`targets/WASP-69b/wasp69b_mie_catalog_layer_by_layer_R1000_optimal_estimation.yaml` for a
deferred OE analysis after a completed MultiNest run has been inspected. It
retrieves one temperature at each of the 80 pressure-layer centres. Run it
with `run_oe_from_nested.py`; the ordinary YAML runner does not perform the
completed-result handoff. Its explicit smoothing prior uses a 250 K marginal
standard deviation and an exponential 1.5-dex log-pressure correlation length.

The short scenario files use `extends` to inherit the common target physics,
opacity, source-data paths, and sampler defaults. ROBERT resolves and validates
the complete configuration before a run; `create_run_directory.py` writes that
resolved configuration into the self-contained run directory.

`targets/WASP-69b/wasp69b_cloud_free_R1000.yaml` is retained as the complete,
standalone WASP-69b cloud-free native baseline that the named WASP-69b defaults
extend.

The matched WASP-69b/WASP-80b regional-emission MultiNest matrix is documented
under [`targets/README.md`](targets/README.md). It creates the simple
`WASP-69b/<model>` and `WASP-80b/<model>` run layout and covers clear,
one-region cloudy, diluted cloudy, and two-region cloudy models. The targeted
matrix explicitly includes the six published overlap-average bins for
WASP-69b.

The Mie configurations are higher-dimensional tests. Begin with the cloud-free and
fixed-catalogue cloud cases; use the direct-n/k cases after the opacity cache,
MPI launch, and fixed-material baseline have been checked.

All defaults use only independent likelihood terms: the WASP-69b overlap
average is excluded when its two parent modes are fitted. MIRI offsets are
also disabled in the baseline configurations; enable one only as a separate
calibration-sensitivity test.

For new projects, start from `examples/TEMPLATE_all_supported_options.yaml`. It groups
the editable inputs into system, data, atmosphere, cloud, opacity/RT, priors,
sampler, plotting, and one top-level paths section. The active block is a valid
cloud-free retrieval; commented alternatives show the supported one-region,
diluted, two-region, chemistry, cloud, and prior modes.
