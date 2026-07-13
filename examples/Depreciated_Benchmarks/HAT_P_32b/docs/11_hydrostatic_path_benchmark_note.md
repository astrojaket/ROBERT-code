# Hydrostatic Path Geometry Benchmark Note

Date: 2026-07-05

ROBERT now has an explicit hydrostatic radius/path-geometry object for emission
RT. It computes layer-edge and layer-center radii from hydrostatic balance,
anchored by a reference radius and reference pressure, then can provide
spherical shell path-length factors for each emission angle.

## Implementation

- `HydrostaticPathGeometry`: stores pressure grid, gravity, scale heights,
  edge radii, center radii, and reference-radius metadata.
- `hydrostatic_path_geometry(...)`: builds that object from an `AtmosphereState`,
  gravity, `reference_radius_m`, and `reference_pressure`.
- `solve_emission(..., path_geometry=...)`: can now use spherical
  shell slant paths instead of the default plane-parallel secant path.
- The single-scattering diagnostic now computes the incoming stellar-beam
  attenuation from the stellar path factors rather than reusing the emission
  path.
- The HAT-P-32b benchmark script exposes:
  - `ROBERT_HAT_P_32B_PATH_GEOMETRY=plane_parallel|hydrostatic_spherical`
  - `ROBERT_HAT_P_32B_ECLIPSE_RADIUS=reference|top|bottom`

The benchmark default remains `plane_parallel`, because that is still the
closest match to the external HAT-P-32b emission output.

## HAT-P-32b Radius Grid

With the HAT-P-32b benchmark atmosphere and `R_p = 1.98 R_JUP` anchored at
100 bar, ROBERT computes:

| Quantity | Value |
| --- | ---: |
| bottom radius | `1.408241181e8 m` |
| reference radius at 100 bar | `1.415541600e8 m` |
| top radius | `1.680707610e8 m` |
| atmosphere thickness | `2.724664291e7 m` |
| median scale height | `1.645191261e6 m` |

That large geometric extension makes spherical limb visibility a major effect.

## Benchmark Matrix

All runs used FastChem, six available R1000 `.kta` species, random overlap, CIA,
and Rayleigh extinction.

| Path model | Disc geometry | Source function | Median model [ppm] | Median residual [ppm] | RMSE [ppm] | Max abs residual [ppm] |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| plane-parallel | Gauss-Legendre thermal disc | thermal Planck | 981.11 | 12.60 | 15.20 | 28.94 |
| plane-parallel | Lobatto phase disc | thermal Planck | 977.39 | 8.93 | 11.74 | 24.22 |
| plane-parallel | Lobatto phase disc | thermal + single scattering | 977.44 | 15.83 | 51.88 | 140.67 |
| hydrostatic spherical | Gauss-Legendre thermal disc | thermal Planck | 873.47 | -65.99 | 139.99 | 522.45 |
| hydrostatic spherical | Lobatto phase disc | thermal Planck | 837.44 | -131.51 | 258.84 | 595.16 |
| hydrostatic spherical | Lobatto phase disc | thermal + single scattering | 837.46 | -131.49 | 261.79 | 595.16 |

## Interpretation

The hydrostatic spherical path model is useful infrastructure, but it is not the
geometry assumption that best matches this HAT-P-32b benchmark. The likely
reason is that the external benchmark's emission disc/path treatment behaves
closer to a plane-parallel secant path than to a spherical shell visibility
model tied to the 100-bar reference radius.

For now, use the spherical path mode as a diagnostic and for future cloud,
surface, transmission, or phase-curve geometry experiments. Keep the
plane-parallel Lobatto phase case as the current best HAT-P-32b thermal
emission comparison.
