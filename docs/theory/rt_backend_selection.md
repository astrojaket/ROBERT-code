# Choosing a Radiative-Transfer Backend

The observable geometry, scattering optical depth, phase function, and required
accuracy determine the RT method. Cloud complexity and solver order are separate
choices: a high-order solver cannot recover information discarded by a grey
cloud model, and a detailed Mie cloud is not used faithfully by a solver that
only retains one asymmetry parameter.

## Science-case guide

| Science case | Appropriate fast path | Science path | Reference validation path |
|---|---|---|---|
| Clear thermal emission | Absorbing formal solution | Same formal solution with correlated-k gas and CIA | Line-by-line or independently implemented formal solution |
| Cloudy thermal emission, low scattering optical depth | Absorbing solution plus an explicitly labelled first-order diagnostic | SH4 when cloud scattering changes the contribution function | 16–32+ stream DOM or matrix-operator solver |
| Cloudy thermal emission, moderate or thick scattering | Toon hemispheric mean for exploratory runs | SH4/P3 with physical phase moments and delta-M treatment for a narrow forward peak | High-stream DOM, doubling/adding, or NEMESIS-style matrix operator |
| Transmission, absorption-dominated limb | Spherical slant-path extinction | Spherical refracting geometry where needed, correlated-k/line-by-line extinction | Independent chord integration or line-by-line transit solver |
| Transmission with optically thin aerosol scattering | Extinction baseline plus an explicitly labelled single-scattering sensitivity test | Multiple-scattering transit treatment if scattered light enters the stellar beam/aperture | Monte Carlo or high-stream spherical/pseudo-spherical solver |
| Transmission through thick or strongly forward-scattering cloud | Extinction provides a conservative baseline, not a complete solution | Multiple scattering with finite stellar angular size and observation aperture | Monte Carlo or validated high-order transit solver |
| Reflected light | Single scattering only for optically thin diagnostics | SH4 or higher-order multiple scattering with direct stellar beam and phase geometry | High-stream DOM, adding-doubling, or Monte Carlo |
| Polarized reflected light | None in the current scalar ROBERT RT | Vector RT using the scattering matrix | Validated vector DOM/adding-doubling/Monte Carlo |

Single scattering is therefore not a general “simple cloud” backend. It is an
optical-depth expansion whose validity must be demonstrated. Reflected-light
spectra normally require multiple scattering because the observable is made of
scattered photons. Thermal emission can use the absorbing formal solution when
scattering is negligible, but an optically thick cloud with appreciable
single-scattering albedo requires multiple scattering even if the cloud opacity
itself is grey.

## Cloud optical-property guide

### Simple retrieval cloud

A grey or power-law cloud can retrieve broad continuum effects with a small
parameter set. It should still specify extinction optical depth, single-
scattering albedo, and a phase-function model. If only asymmetry `g` is
available, ROBERT's SH4 backend uses an explicitly recorded Henyey–Greenstein
closure, `chi_l = (2l+1) g^l`, and delta-M scaling by default. This is a model
assumption, not a unique consequence of `g`.

The Toon hemispheric-mean backend is useful for rapid scans and regression
checks. It is not the preferred final backend for angle-resolved spectra from
forward-scattering clouds: the controlled `omega=0.9`, `g=0.6` benchmark shows
sub-percent disk agreement with PICASO Toon but a 35% maximum point-radiance
difference, while matched SH4 implementations agree to about 3 parts per
million.

### Microphysical cloud

A science cloud model should provide layer- and wavelength-resolved extinction,
single-scattering albedo, and enough Legendre moments to represent the computed
Mie, aggregate-particle, or laboratory phase function. Particle size
distribution, composition, shape assumptions, and mixing must be retained in
provenance. Reducing such a phase function to one `g` value loses information
that matters most for reflected light and strongly forward-scattering transit
geometries.

SH4 retains moments through degree three and is a practical retrieval backend.
It is still a finite angular expansion. Extreme forward peaks, sharp opposition
features, polarization, or precision claims near the solver error require a
higher-order reference calculation. Delta-M scaling improves a truncated
expansion by treating the unresolved forward peak separately; it does not turn
SH4 into an exact phase-function solution.

## Current ROBERT status

- Absorbing thermal formal solution: implemented and tested against the Rutten
  formal solution.
- Direct-beam single scattering: implemented as a diagnostic source treatment.
- Toon hemispheric-mean thermal multiple scattering: implemented and checked
  against PICASO Toon.
- Thermal SH4/P3: implemented with linear Planck sources, a banded multilayer
  boundary solve, physical or HG phase moments, and optional delta-M scaling.
  The high-level emission backend mixes supplied cloud and Rayleigh moments by
  scattering optical depth; the homogeneous-sphere Mie model supplies moments
  through degree four directly.
- Absorption-dominated spherical transmission: implemented with exact shell
  chords and correlated-k impact-area integration.
- Transmission multiple scattering: not implemented.
- Reflected-light multiple scattering: not implemented; the SH moment operator
  is designed to be reused, but the direct-beam source and reflected boundary
  problem still require their own implementation and validation.
- High-stream reference: not yet integrated. It remains required before a
  production science claim across the full cloud-property domain.

The thermal SH4 formulation follows [Rooney, Batalha & Marley
(2024)](https://doi.org/10.3847/1538-4357/ad05c5). The transfer convention and
formal solution follow [Rutten
(2003)](https://robrutten.nl/rrweb/rjr-edu/coursenotes/rutten_rtsa_notes_2003.pdf),
and the cloudy-emission science limits follow [Taylor et al.
(2021)](https://doi.org/10.1093/mnras/stab1854).
