# v0.3 RT Benchmark Audit

Date: 2026-07-05

Status: superseded for hydrostatic path results by
`docs/review/11_hydrostatic_path_benchmark_note.md`.

This note records the current ROBERT state after the v0.3 forward-model
foundation work, compares it to the original staged plan, and keeps the
historical source inspiration in documentation rather than runtime code strings
or user-facing metadata.

## Executive Assessment

ROBERT is on the right track for an emission-retrieval code. The implementation
has moved beyond the initial skeleton into a modular forward-model foundation:
atmosphere, P-T profiles, chemistry, opacity import, correlated-k mixing,
optical-depth diagnostics, geometry, and a NumPy emission reference solver are
now connected through typed objects.

The current benchmark results suggest the dominant remaining mismatch is not
the broad thermal-emission physics. The best current HAT-P-32b comparison is
the phase-aware Lobatto geometry without the first scattering-source term,
which reduces the full-spectrum RMSE from 15.20 ppm to 11.74 ppm. The optional
first single-scattering source is useful as a diagnostic, but it currently
over-predicts the short-wave flux in this benchmark and should remain off for
thermal-emission parity runs.

## What Is Done

- Core domain objects: pressure grids, spectral grids, spectra, planet/star,
  observations, config containers, and likelihood helpers.
- Temperature profiles: isothermal, tabulated input, spline retrieval profile,
  Madhusudhan & Seager (2009), and Parmentier & Guillot (2014) style profiles.
- Chemistry: constant/free chemistry, H2/He background filling, composition
  mean molecular weight, fixed mean molecular weight, and an optional FastChem
  equilibrium adapter.
- Opacity: ExoMol/HITRAN-style inspectors, `.kta` reader, non-finite runtime
  floor policy for incomplete tables, ROBERT `.npy`/`.npz` archive helpers,
  correlated-k native-grid evaluation, and interpolation support.
- Gas mixing: summed same-g and random-overlap correlated-k optical-depth
  combination, with Numba acceleration for the hot random-overlap path.
- Extinction: CIA optical-depth helper, H2/He Rayleigh optical-depth helper,
  and additive layer optical-depth containers.
- RT geometry: normal emission, Gauss-Legendre thermal disc, and Lobatto
  projected-disc phase geometry.
- RT solver: NumPy clear-sky thermal-emission reference solver with Planck
  source integration, layer contribution diagnostics, disc averaging, eclipse
  depth normalization, and optional first-order direct-beam single scattering.
- Diagnostics: blackbody sanity plots, tau/weighting plots, opacity benchmark
  plots, HAT-P-32b emission benchmark plots, and JSON metric summaries.

## Original Plan Comparison

The original v0.3 plan only required a minimal non-retrieval forward model,
simple atmosphere construction, placeholder opacity, and tested pipeline
wiring. ROBERT has completed that and pulled several later roadmap items
forward because the HAT-P-32b benchmark needed them before retrieval work:

| Area | Original v0.3 plan | Current status |
| --- | --- | --- |
| Atmosphere | Isothermal/simple state | Done, plus tabulated and retrieval-ready P-T profiles |
| Chemistry | Constant/free chemistry | Done, plus background fill, MMW, and FastChem adapter |
| Opacity | Placeholder interface | Done, plus `.kta` import/archive and correlated-k evaluator |
| Gas mixing | Not scheduled until later | Done for summed same-g and random overlap |
| CIA/Rayleigh | Later RT work | First optical-depth helpers done |
| RT | Placeholder/reference only | NumPy thermal-emission reference solver done |
| Scattering | Later RT work | First diagnostic single-scattering source done |
| Plots | Basic blackbody/benchmark plots | Done across blackbody, opacity, tau, RT, geometry, and scattering |

## Benchmark Matrix

All three runs used the local HAT-P-32b benchmark CSV, P-T CSV, six available
R1000 `.kta` species, FastChem chemistry from the local FastChem install, random
overlap, CIA, and Rayleigh extinction.

| Case | Geometry | Source function | Runtime [s] | Median model [ppm] | Median benchmark [ppm] | Median residual [ppm] | RMSE [ppm] | Max abs residual [ppm] |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | Gauss-Legendre thermal disc | thermal Planck | 4.19 | 981.11 | 968.29 | 12.60 | 15.20 | 28.94 |
| Phase geometry | Lobatto phase disc | thermal Planck | 4.18 | 977.39 | 968.29 | 8.93 | 11.74 | 24.22 |
| Diagnostic scattering | Lobatto phase disc | thermal + single scattering | 4.39 | 977.44 | 968.29 | 15.78 | 42.98 | 114.99 |

Generated plots:

- `examples/outputs/hat_p_32b_emission_rt_benchmark/hat_p_32b_clear_sky_emission_rt.png`
- `examples/outputs/hat_p_32b_emission_rt_benchmark/hat_p_32b_clear_sky_emission_rt_lobatto_phase.png`
- `examples/outputs/hat_p_32b_emission_rt_benchmark/hat_p_32b_clear_sky_emission_rt_lobatto_phase_single_scattering.png`
- `examples/outputs/single_scattering_reference/single_scattering_phase_reference.png`

## Interpretation

The phase-aware Lobatto geometry improves the benchmark without changing the
thermochemistry or opacity inputs, so the next scientific gap to close is
proper path/radius/reference-pressure geometry rather than adding retrieval
machinery immediately.

The first-order scattering source behaves qualitatively as expected by adding a
large short-wave contribution, but it is not the right default for the current
thermal-emission benchmark. It should remain a diagnostic path until ROBERT has
cloud/aerosol optical properties and a more complete scattering solver.

The opacity and gas-mixing boundaries are modular enough to add future opacity
sampling and line-by-line modes. The current API separates storage format,
numerical mode, source metadata, prepared opacity, and RT-facing optical depth,
so additional ExoMol/HITRAN/HITEMP products can be added without rewriting the
thermal-emission solver.

## Provenance Policy

The architecture and benchmark lineage from NEMESIS/NemesisPy remains important
scientific context and is documented in `docs/review/05_lessons_learned_from_nemesis_and_nemesispy.md`.
Runtime code strings, public helper names, and generated ROBERT metadata should
describe ROBERT algorithms and file formats directly. The active code now uses
neutral names such as `KtaTable`, `KtaHeader`, `CiaTable`, `read_cia_table`,
`from_kta`, `lobatto_phase_geometry`, and `kta_binary`.

## Next Build Order

1. Implement hydrostatic height/path geometry and reference-radius-pressure
   handling, then rerun the same HAT-P-32b benchmark matrix.
2. Add cloud/aerosol optical-property containers: extinction, absorption,
   scattering, single-scattering albedo, asymmetry/phase inputs, and provenance.
3. Add a conservative multiple-scattering reference backend, likely two-stream
   first, behind the existing RT interface.
4. Decide the default fast opacity working format from measured `.kta`, `.npy`
   directory, `.npz`, and possible HDF5/Zarr benchmark results.
5. Only after the forward model is benchmark-stable, move back up to instrument
   convolution/binning, run manifests, and sampler-independent retrieval
   problem objects.
