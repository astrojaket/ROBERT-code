# Development roadmap

Status: 2026-09-07. ROBERT is production-ready for its documented and
benchmarked emission, transmission, and retrieval regimes. This roadmap records
extensions, including high-resolution and accelerator work with separate
validation requirements.
See the [repository audit](../review/53_repository_audit_2026-09-07.md) for evidence
and limitations.

## Current position

ROBERT has typed scientific objects, strict schema-v2 YAML, emission and
transmission models, free and equilibrium chemistry, cloud models, instrument
binning, Gaussian likelihoods, optimal estimation, PyMultiNest, and
portable run products. NumPy, Numba, and optional JAX paths exist.

Python APIs also provide line-by-line opacity, high-resolution responses,
filtered observations, covariance likelihoods, and mixed-resolution retrieval
problems. These are not all available through the standard YAML runners.
The WASP-77Ab shared-VMR real-data sampler runs remain deferred to a cluster;
operator checks and fixed-template recovery do not establish their completion.

The package version is `0.3.0`. It does not indicate which scientific regimes
are validated. The old version-by-feature sequence in RFC-0001 is historical;
use the outcome gates below for new work.

## 1. Establish one supported workflow contract

- State which features work through YAML, Python, and target-specific scripts.
- Keep one config-to-problem construction path for shared behavior.
- Test each supported path from input validation to serialized results.
- Define stable and experimental APIs before choosing the next release version.

Done when a new user can run a bundled forward model and small retrieval from
the documented commands, with input identity and limitations in the outputs.

## 2. Reduce coupling at extension points

- PyMultiNest is the sole nested sampler; Optimal Estimation remains a separate
  capability for independent Gaussian errors. Extend OE for detailed sounding
  only with a stated observation model and numerical validation case.
- Keep atmosphere, opacity, RT, response, likelihood, and sampling separate.
- Split large modules by independent responsibility when a change needs it.
- Use the same sampler-facing contract for CPU and accelerator problems.
- Keep target acquisition and observing-program assumptions at the I/O edge.
- Add one tested third-party extension only when a concrete consumer needs it.

Done when a new response or backend does not require copied workflow logic.
A registry for every component family is not a prerequisite.

## 3. Make quality gates match supported paths

- Add incremental static type checks for the stable domain and problem APIs.
- Separate portable tests from native sampler, accelerator, and external-data checks.
- Check public commands and documentation links in CI.
- Run hardware parity and resource checks on the hardware being supported.

Done when skipped optional checks are visible and release evidence identifies
which backends and platforms were exercised. Keep analytic limits, numerical
regressions, and serialization tests. Remove tests that only repeat source text
or duplicate the same behavior.

## 4. Complete the scientific release matrix

- Record independent cloudy calculations with science molecular opacity.
- Test posterior calibration with repeated seeds and evidence stability.
- Complete the deferred shared-VMR HRS/LRS real-data runs.
- State covariance independence, filtering rank, opacity provenance, and
  numerical-resolution assumptions for each case.

Done when each claimed regime has reproducible inputs, tolerances, results,
and a recorded review. An implementation or a passing unit test is not a
scientific validation result.

## Later work

Prioritize new physics by a stated observing need and a validation case.
Candidates include transmission scattering return, refraction, stellar
heterogeneity, calibrated JWST products, disequilibrium chemistry, phase
curves, and reflected light. Do not expand these areas before the existing
workflow and validation contracts are clear.
