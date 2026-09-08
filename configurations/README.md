# Maintained configurations

The maintained user configurations are short, self-contained YAML files in
this directory. Use the file that matches the model and data you want to
test. The complete schema and every supported option are in
[`docs/configuration.md`](../docs/configuration.md).

| File | Use | Required inputs |
| --- | --- | --- |
| [`quickstart.yaml`](quickstart.yaml) | Bundled R=100 emission injection recovery | The generated synthetic observation under `../ROBERT-runs/quickstart-emission-r100/outputs/` |
| [`emission.yaml`](emission.yaml) | WASP-69b cloud-free hot-Jupiter emission at R=1000 | `data/wasp69b_schlawin2024/`, `data/chemistry/fastchem/`, PHOENIX data, and `opacity_data/ktables_exomol/` |
| [`cloudy_emission.yaml`](cloudy_emission.yaml) | WASP-69b PG14 emission with a fixed MgSiO3 Mie cloud | The emission inputs plus `data/optical_constants/exo_skryer/` |
| [`transmission.yaml`](transmission.yaml) | Bundled R=100 hot-Jupiter free-chemistry transmission | The generated synthetic observation under `../ROBERT-runs/hot-jupiter-transmission-r100/outputs/` |
| [`rocky_transmission_clr.yaml`](rocky_transmission_clr.yaml) | L 98-59 b terrestrial transmission with a joint CLR composition prior | `data/observations/l98_59b_bello_arufe2025/` and R=1000 K-tables |
| [`optimal_estimation.yaml`](optimal_estimation.yaml) | WASP-69b cloud-free emission with optimal estimation | The same inputs as `emission.yaml` |
| [`two_region_emission.yaml`](two_region_emission.yaml) | WASP-69b independent hot and cold MgSiO3 emission regions | The same inputs as `cloudy_emission.yaml` |

The quickstart and transmission files use the bundled R=100 table set so that
the validation workflow can run without a high-resolution opacity archive.
For a science run, change `opacity.resolution` to `R1000` and set
`paths.k_table_directory` to the external `opacity_data/ktables_exomol`
directory. The R=1000 WASP files already use that science default.

All writable paths in these templates are under the sibling
`../ROBERT-runs/<run.name>/` directory. This keeps generated observations,
outputs, opacity caches, and scratch files outside the ROBERT source checkout.
Published observations, FastChem inputs, bundled opacities, and external
R=1000 inputs stay at their existing shared paths.

The rocky example is explicitly for **L 98-59 b**. Its loader,
`bello_arufe2025_l9859b`, points to the published b spectrum and must remain
matched to that planet. CLR composition priors require direct MultiNest
sampling; see the prior section in the configuration reference.

All posterior plots use `mediumpurple` by default and request 100 posterior
predictive samples. Run `scripts/create_run_directory.py` to create an
isolated run directory with resolved paths and writable output directories.
The distinct PICASO and six-gas validation inputs remain under
`tests/fixtures/configurations/` for regression tests.
