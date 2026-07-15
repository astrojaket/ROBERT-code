# Independent transmission geometry audit

## Scope

This audit investigates the few-ppm ROBERT--PICASO transmission residual
without treating PICASO as ground truth or changing ROBERT to reproduce it. It
holds the evaluated opacity fixed and changes one numerical convention at a
time. All ROBERT calculations used the `robert-exoplanets` conda environment;
the external calculation used PICASO 3.2.2 and the official R=15000 opacity
database at stride 5.

The first matched-optical-depth benchmark saved molecular/CIA and cloud
optical depths, but omitted PICASO's Rayleigh optical depth and exact
density/column profiles. The independent runner now exports the complete
`taugas + tauray + taucld` extinction budget together with PICASO's level
radii, pressures, temperatures, layer mean molecular weights, and column
masses. The corrected result supersedes the preliminary 3.31--6.33 ppm cloud
effect range in review 40.

## Reproduction check

An independent vectorized implementation of PICASO's Brown (2001) level sum
was evaluated from the exported arrays. It reproduces PICASO to
4.72e-7 ppm RMS for clear, deck, haze, and deck+haze spectra. Cloud-effect
differences are below 4.4e-9 ppm. This verifies that the audit has captured the
relevant PICASO transmission algorithm rather than merely constructing a
similar approximation.

## Algorithm decomposition

The two codes make two different finite-layer choices:

1. PICASO converts vertical layer optical depth to slant extinction using gas
   density at a bounding pressure level. ROBERT distributes the integrated
   vertical optical depth uniformly through the constant-property spherical
   shell.
2. PICASO evaluates one tangent ray per pressure level and uses a rectangular
   annulus sum. ROBERT integrates multiple impact radii in every annulus with
   Gauss--Legendre quadrature.

At 80 layers, the absolute RMS differences from PICASO are:

| Extinction within layer | Annulus integration | Clear | Deck | Haze | Deck+haze |
|---|---|---:|---:|---:|---:|
| PICASO level density | PICASO level rectangle | 0.00000047 ppm | 0.00000047 ppm | 0.00000047 ppm | 0.00000047 ppm |
| PICASO level density | ROBERT shell Gaussian | 17.10 ppm | 16.71 ppm | 17.09 ppm | 16.68 ppm |
| ROBERT uniform shell | PICASO level rectangle | 18.20 ppm | 19.27 ppm | 18.47 ppm | 18.68 ppm |
| ROBERT uniform shell | ROBERT shell Gaussian | 1.90 ppm | 4.24 ppm | 2.19 ppm | 3.25 ppm |

The two approximately 17--18 ppm changes have opposite signs and largely
cancel when both ROBERT conventions are used. The remaining cloud-effect RMS
differences are 3.49 ppm for the deck, 0.67 ppm for haze, and 2.43 ppm for
deck+haze. This is why the residual is scientifically relevant but does not
look like a single systematic radius offset.

ROBERT's annulus quadrature itself is not limiting: order 8 differs from order
24 by 0.0052 ppm RMS in absolute depth and at most 0.00026 ppm in cloud effect.
The Gaussian annulus integral is the more converged numerical treatment for
the stated piecewise-constant shell model. Whether uniform shell extinction or
level-density placement better approximates a continuously varying atmosphere
should be decided with an analytic or very-high-resolution reference, not by
forcing agreement with PICASO's finite-layer result.

## Radius, gravity, and pressure anchor

Consistently remapping the same physical pressure-radius curve from a 10 bar
anchor to 1 or 0.1 bar changes neither the radius grid nor the spectrum. By
contrast, assigning the same numerical radius to 1 bar that was defined at
10 bar shifts the atmosphere by about 1530 km and the clear spectrum by about
770 ppm. A wrong reference pressure therefore produces a large near-constant
offset, but it is not the cause of the corrected few-ppm residual.

Reconstructing PICASO's sequential inverse-square profile with ROBERT's layer
representation changes edge radii by only 457 m RMS. Constant gravity changes
them by roughly 380 km RMS and produces about 193 ppm clear-spectrum RMS error.
Variable inverse-square gravity is therefore necessary across this extended
atmosphere; the small difference between the two inverse-square
discretizations is secondary.

## Interpretation and next independent reference

Perfect finite-resolution agreement between independently designed codes is
not the objective. Exact agreement is expected only after the physical and
numerical contracts are deliberately made identical, as demonstrated by the
PICASO reproduction check. ROBERT should retain its converged spherical
annulus integral and explicit constant-property shell semantics unless an
independent accuracy test shows a better choice.

The next verification should use a prescribed continuous isothermal or
non-isothermal atmosphere with analytic opacity laws. Compute a very fine
direct line-of-sight integral as the reference, then measure ROBERT and PICASO
convergence independently with layer count. That will determine which
finite-layer extinction placement approaches the continuous solution faster
and whether the residual matters at a realistic JWST binning and covariance,
rather than only in native-model ppm.

## Artifacts

- `examples/outputs/transmission_geometry_audit/transmission_geometry_audit.json`
- `examples/outputs/transmission_geometry_audit/transmission_geometry_audit.png`
- `examples/outputs/transmission_geometry_audit/transmission_algorithm_audit.png`
- `examples/outputs/transmission_geometry_audit/transmission_geometry_audit_spectra.npz`
