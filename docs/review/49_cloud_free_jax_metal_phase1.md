# Cloud-free correlated-k JAX Metal phase 1

Date: 2026-08-03

## Superseded benchmark notice

The original benchmark recorded in this file used an isothermal atmosphere
with an equal-temperature blackbody lower boundary.  In LTE clear emission,
that configuration telescopes to the same Planck spectrum independently of
gas opacity.  Its previously reported spectrum, likelihood, grid-posterior,
and evidence agreement therefore did **not** validate correlated-k opacity,
random overlap, or abundance retrieval.  Those scientific claims have been
withdrawn.

The old timing was real for the executed graph—roughly 1.10 s per warm Metal
call versus 3.39 ms for the specialized CPU path—but it is retained only as a
historical performance diagnosis of JAX Metal resort/rebin control flow.  It
must not be cited as a science-parity result.

## Corrections now in the code

The cloud-free correlated-k model now:

- retrieves the five Parmentier-Guillot 2014 parameters followed by log VMRs;
- evaluates temperature independently at layer centers and edges;
- uses four-point Gauss-Legendre disk integration rather than one normal ray;
- enforces exact parameter ordering and provider interpolation semantics;
- validates shared pressure, temperature, spectral, g-ordinate, and unit grids;
- records opacity source identifiers and checksums;
- exposes a manifest-compatible likelihood; and
- retains the one-proposal unified-memory guard.

The corrected non-isothermal tests perturb every gas and require nonzero CPU
and JAX spectral responses.  Both bounded cloud-free correlated-k and
opacity-sampling graphs execute successfully on the real Apple Metal device.

No corrected correlated-k performance/posterior benchmark is claimed here.
Development was deliberately refocused on opacity sampling, which removes
random-overlap sorting and is a substantially better accelerator workload.
The measured replacement study is
`docs/review/50_jax_opacity_sampling_accelerator.md`.

## Current decision

Correlated-k Metal remains an isolated experimental correctness path, not a
speed option.  The CPU implementation is unchanged and remains the reference.
Cloud scattering is outside this phase.
