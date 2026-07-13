# Default WASP configurations

Every file here is a validated default for the YAML runners. The defaults use
the faster R1000 ExoMol K-tables. To make an R15000 counterpart, copy the
selected file, change `opacity.resolution: R15000`, give it a new `run.name`,
and create a new run directory. Do not mix R1000 and R15000 checkpoints or
prepared opacity caches.

| Target | Scenario | Default YAML |
| --- | --- | --- |
| WASP-69b | Cloud-free, native F322W2/F444W/LRS, PG14 | `wasp69b_cloud_free_native_pg14_R1000.yaml` |
| WASP-69b | Cloud-free, NIRCam F322W2/F444W, PG14 | `wasp69b_cloud_free_nircam_pg14_R1000.yaml` |
| WASP-69b | Cloud-free, native modes, retrieved isothermal T-P | `wasp69b_cloud_free_native_isothermal_R1000.yaml` |
| WASP-69b | Fixed MgSiO3 catalogue Mie cloud, PG14 | `wasp69b_mie_catalog_pg14_R1000.yaml` |
| WASP-69b | Retrieved Mie n/k cloud, PG14 | `wasp69b_mie_direct_nk_pg14_R1000.yaml` |
| WASP-80b | Cloud-free, native F322W2/F444W/LRS, PG14 | `wasp80b_cloud_free_native_pg14_R1000.yaml` |
| WASP-80b | Cloud-free, NIRCam F322W2/F444W, PG14 | `wasp80b_cloud_free_nircam_pg14_R1000.yaml` |
| WASP-80b | Cloud-free, native modes, retrieved isothermal T-P | `wasp80b_cloud_free_native_isothermal_R1000.yaml` |
| WASP-80b | Fixed MgSiO3 catalogue Mie cloud, PG14 | `wasp80b_mie_catalog_pg14_R1000.yaml` |
| WASP-80b | Retrieved Mie n/k cloud, PG14 | `wasp80b_mie_direct_nk_pg14_R1000.yaml` |

The short scenario files use `extends` to inherit the common target physics,
opacity, source-data paths, and sampler defaults. ROBERT resolves and validates
the complete configuration before a run; `create_run_directory.py` writes that
resolved configuration into the self-contained run directory.

`wasp69b_cloud_free_R1000.yaml` is retained as the complete, standalone original
WASP-69b cloud-free native baseline that the named WASP-69b defaults extend.

The Mie configurations are higher-dimensional tests. Begin with the cloud-free and
fixed-catalogue cloud cases; use the direct-n/k cases after the opacity cache,
MPI launch, and fixed-material baseline have been checked.

All defaults use only independent likelihood terms: the WASP-69b overlap
average is excluded when its two parent modes are fitted. MIRI offsets are
also disabled in the baseline configurations; enable one only as a separate
calibration-sensitivity test.

For new projects, start from `TEMPLATE_all_supported_options.yaml`. It groups
the editable inputs into system, data, atmosphere, cloud, opacity/RT, priors,
sampler, and housekeeping sections. The active block is a valid cloud-free retrieval;
commented alternatives show the currently supported optional modes and priors.
