# Shared-State Scientific Regression and Code Audit

Date: 2026-07-12

## Scope

This audit treats the complete working tree as the scientific baseline. It
focuses on the new multi-dataset flow in which one atmospheric state is built
per parameter vector and then evaluated through an arbitrary number of
mode-specific correlated-k opacity and radiative-transfer models.

The active WASP-69b UltraNest run was not stopped, restarted, or modified.
Validation used the Python 3.12 `robert-exoplanets` environment.

## Result

No physics regression was found. All 16 `examples/benchmark_*.py` programs
completed successfully. Fourteen persisted benchmark reports were snapshotted
before the audit and compared after execution. After excluding timing and
runtime-provenance fields, there were zero changed scientific fields.

The strengthened four-mode WASP-69b benchmark independently builds the
atmosphere once per instrument for its comparison path and once in total for
the default shared path. At the benchmark parameter vector:

- all four mode spectra are bit-for-bit equal;
- every per-mode maximum absolute spectral difference is `0.0`;
- both paths return exactly the same log likelihood,
  `-61339.66439309031`;
- the separate native-spectrum-then-bin diagnostic returns
  `-61446.289006481966`, as expected because integrating an evaluated flux
  spectrum is not equivalent to correlated-k recompression before RT.

The default shared path therefore changes orchestration only. It does not
replace, average, interpolate, or approximate any atmospheric, opacity, CIA,
or RT result.

## Shared-State Flow Review

`build_multi_dataset_emission_model` requires every mode configuration to use
the same temperature profile, chemistry model, mean-molecular-weight model,
pressure grid, CIA tables, geometry, planet, star, mean molecular weight, and
emission configuration. Each mode retains its own opacity provider and
spectral grid. The resulting `MultiDatasetEmissionForwardModel` also verifies
that every prepared model holds the same `AtmosphereBuilder` instance and
requires the same parameter tuple.

For each likelihood call the shared model:

1. validates one parameter vector;
2. builds one immutable atmospheric state;
3. sends that exact state to every mode-specific opacity evaluation and RT
   solve;
4. returns an immutable name-to-spectrum mapping.

Pressure-grid identity is preferred, with exact edge-and-centre equality as a
safe fallback. There is no tolerance-based acceptance that could hide a
physical mismatch.

The native-spectrum response path has the deliberately explicit
`NativeSpectrumMultiDatasetForwardModel` name and remains a plotting and
diagnostic tool. The old ambiguous `MultiDatasetForwardModel` and
`MultiDatasetPrediction` exports are absent. The deliberately inefficient
repeated-atmosphere implementation is local to the WASP performance benchmark
and cannot be selected accidentally by retrieval code.

## Bloat and Deprecation Review

The audit searched source, examples, tests, and documentation for deprecated,
legacy, compatibility, placeholder, stub, TODO, and unimplemented markers;
deprecated NumPy/SciPy/Python aliases; unresolved or duplicate exports; and
references to superseded multi-dataset names.

One confirmed dead compatibility remnant was removed:

- `Observation.validate()` was a v0.1 no-op with no repository callers.
  `Observation.__post_init__` already performs all validation, so removing the
  method deletes no validation or physics.

The remaining compatibility paths are intentional and tested scientific data
or numerical interfaces, notably early opacity-archive reading and legacy RT
quadrature inputs. They were retained because removing them would discard real
supported behavior rather than bloat.

The shared model's `validated_parameters` and `evaluate_atmosphere` methods are
both live: the standalone single-mode call uses both, while the multi-mode
orchestrator uses them to separate the one-time atmosphere build from each
mode solve. No duplicate or unreachable shared-state implementation remains.

## Performance Findings

A warmed `cProfile` capture of the WASP-69b paths contained 14.555 seconds of
profiled work. The principal cumulative costs were:

| Component | Cumulative time | Approximate share |
| --- | ---: | ---: |
| Random-overlap gas mixing | 8.415 s | 58% |
| Clear-emission RT solve | 2.100 s | 14% |
| CIA optical depth | 1.832 s | 13% |
| Opacity pressure-temperature interpolation | 1.494 s | 10% |

The shared-state orchestration and parameter validation do not appear among
the significant profile entries. In the final two-repeat four-mode check, the
independent path took a median `0.4180 s` and the shared path `0.3480 s`, a
`1.20x` speedup. Timing is advisory because the retrieval and other processes
were active; the exact spectral and likelihood comparisons are deterministic.

The next optimization target should remain random overlap. Its cost grows with
species, layers, wavelengths, and g ordinates and now dominates the laptop
forward model. The next isolated opportunity is CIA interpolation: 126 CIA
evaluations invoked the coefficient interpolator 10,080 times and ultimately
called `numpy.interp` 92,680 times. Vectorizing or precomputing its spectral
and temperature interpolation indices is a promising accuracy-preserving next
step. Opacity interpolation is third. Further changes should keep the NumPy
reference implementations and add exact or tightly bounded parity tests before
changing the accelerated path.

## Verification Matrix

- 16/16 benchmark programs passed.
- 14 persisted deterministic reports had zero non-timing scientific changes.
- WASP-69b independent/shared spectra: exact equality in all four modes.
- WASP-69b independent/shared likelihood: exact equality.
- Random-overlap benchmark maximum NumPy-reference difference:
  `2.7711166694643907e-13` in the largest six-species case.
- Test suite: 281 passed; branch coverage 76%, above the configured 70% floor.
- Ruff lint: passed for source, tests, and examples.
- Ruff formatting: all audit-touched Python files passed. The repository-wide
  format check still reports older baseline files that were not reformatted in
  this audit to avoid an unrelated bulk diff.
- Bytecode compilation: passed for source and examples.
- Public API: 183 exports resolve; no duplicates; superseded multi-dataset
  aliases are absent.
- Distribution: sdist and wheel built successfully.
- `git diff --check`: passed.

The only emitted warnings are third-party PyParsing deprecations through
Matplotlib and an Intel OpenMP `omp_set_nested` notice. ROBERT does not call
either deprecated API.

## Conclusion

The shared atmospheric state is now the unambiguous default retrieval flow,
and the available regression evidence shows that it preserves prior benchmark
physics exactly. The audit removed one harmless no-op compatibility method and
strengthened the production-scale benchmark. Further meaningful laptop
speedups will require work inside random-overlap mixing, CIA interpolation,
and opacity interpolation rather than more orchestration changes.
