# Default WASP configurations

Every file here is a validated default for the YAML runners. The defaults use
the faster R1000 ExoMol K-tables. To make an R15000 counterpart, copy the
selected file, change `opacity.resolution: R15000`, give it a new `run.name`,
and create a new run directory. Do not mix R1000 and R15000 checkpoints or
prepared opacity caches.

| Target | Scenario | Default YAML |
| --- | --- | --- |
| WASP-69b | Clear, native F322W2/F444W/LRS, PG14 | `wasp69b_clear_native_pg14_R1000.yaml` |
| WASP-69b | Clear, NIRCam including overlap average, PG14 | `wasp69b_clear_nircam_pg14_R1000.yaml` |
| WASP-69b | Clear, native modes, retrieved isothermal T-P | `wasp69b_clear_native_isothermal_R1000.yaml` |
| WASP-69b | Fixed MgSiO3 catalogue Mie cloud, PG14 | `wasp69b_mie_catalog_pg14_R1000.yaml` |
| WASP-69b | Retrieved Mie n/k cloud, PG14 | `wasp69b_mie_direct_nk_pg14_R1000.yaml` |
| WASP-80b | Clear, native F322W2/F444W/LRS, PG14 | `wasp80b_clear_native_pg14_R1000.yaml` |
| WASP-80b | Clear, NIRCam F322W2/F444W, PG14 | `wasp80b_clear_nircam_pg14_R1000.yaml` |
| WASP-80b | Clear, native modes, retrieved isothermal T-P | `wasp80b_clear_native_isothermal_R1000.yaml` |
| WASP-80b | Fixed MgSiO3 catalogue Mie cloud, PG14 | `wasp80b_mie_catalog_pg14_R1000.yaml` |
| WASP-80b | Retrieved Mie n/k cloud, PG14 | `wasp80b_mie_direct_nk_pg14_R1000.yaml` |

The short scenario files use `extends` to inherit the common target physics,
opacity, source-data paths, and sampler defaults. ROBERT resolves and validates
the complete configuration before a run; `create_run_directory.py` writes that
resolved configuration into the self-contained run directory.

`wasp69b_clear_R1000.yaml` is retained as the complete, standalone original
WASP-69b clear-native baseline that the named WASP-69b defaults extend.

The Mie configurations are higher-dimensional tests. Begin with the clear and
fixed-catalogue cloud cases; use the direct-n/k cases after the opacity cache,
MPI launch, and fixed-material baseline have been checked.
