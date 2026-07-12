# Cloud-agnostic refractive-index emission model

## Scientific boundary

ROBERT now separates material description from cloud radiative transfer. A
cloud material is represented by its complex refractive index

`m(lambda) = n(lambda) + i k(lambda)`,

with positive `k` denoting absorption. Material names are catalogue metadata,
not branches in the forward model. This supports both known-condensate runs
and direct retrieval of optical constants for comparison with laboratory
materials after inference.

The retrieval representation uses increasing wavelength nodes with independent
`n_i` and `log10(k_i)` parameters. Interpolation is linear in log wavelength;
positive `k` is interpolated logarithmically. The initial implementation does
not impose Kramers-Kronig consistency because finite wavelength coverage makes
that constraint prior-dependent. A future optional causal prior may derive
`n(lambda)` from `k(lambda)` plus a reference index and explicit assumptions
about opacity outside the observed band.

## Particle and vertical model

`mie_efficiencies` implements the Lorenz-Mie solution for homogeneous spheres,
with an analytic Rayleigh limit for very small size parameters. The scalar Mie
phase function is integrated to obtain its Legendre moments through degree
four. A lognormal number distribution is integrated over six standard
deviations in log radius with Gauss-Legendre quadrature. Its configured radius
is the area-weighted effective radius, `r_eff = <r^3>/<r^2>`, rather than the
geometric-mean radius. Cross sections are divided by the distribution-averaged
particle mass to obtain condensate mass extinction and scattering coefficients.

The initial WASP-69b retrieval uses `sigma_g = 1`, i.e. monodisperse spheres.
This is deliberate: tests over broad distributions show that the number of
radius quadrature points required depends strongly on radius, width,
wavelength, and refractive index. Any future retrieval with `sigma_g > 1` must
demonstrate convergence for its own prior domain.

For condensate mass fraction `q_cond` per unit bulk atmospheric mass, each
layer has

`delta_tau_ext(lambda) = kappa_ext(lambda) q_cond delta_pressure / gravity`.

The retrieval can fit `q_cond`, effective particle radius, optional geometric
width, and optional cloud-top/base pressure. Particle density remains an
explicit configured material property because spectra usually constrain the
combination of density and condensate abundance rather than both independently.

## Optical-constant catalogue

`data/optical_constants/exo_skryer/` is a verbatim snapshot of Exo Skryer's
`nk_data` directory at commit
`abde13e1e39753f0528e271227decec503211c64`. It is kept outside the MIT Python
package data and carries the source repository's AGPL-3.0 licence. Individual
file headers retain laboratory citations. `OpticalConstantsCatalog` selects
materials by filename stem and records the source checksum when loading.

Source irregularities are handled transparently rather than rewritten:

- repeated wavelengths at joins are averaged and counted in metadata;
- the `TiO2` declared row-count mismatch is recorded;
- the artificial `White` pseudo-species is hidden;
- `Vacuum` remains as a numerical reference.

## Validation and current boundary

Mie extinction, scattering, asymmetry, and exact scalar phase moments now feed
ROBERT's cloud optical-property interface. Cloud and molecular Rayleigh moments
are scattering-optical-depth weighted before SH4. SH4 retains degrees zero
through three and uses the degree-four Mie moment to define its delta-M forward
fraction. It only falls back to a Henyey-Greenstein closure for an external
cloud object that supplies `g` but no phase moments.

Efficiencies, asymmetry, and moments were independently checked against
`miepython 3.2.0` from Rayleigh particles through size parameter 20. The
detailed numerical record is in
[`25_mie_cloud_validation.md`](25_mie_cloud_validation.md).

This is a validated homogeneous-sphere thermal-emission model, not a universal
cloud solver. Particle shape, porosity, composition mixing, polarization, and
non-spherical phase matrices are not represented. SH4 is a four-term angular
method; a high-stream comparison over the posterior cloud domain remains
required before claiming accuracy for extreme forward peaks.
