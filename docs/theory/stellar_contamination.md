# Transit light source effect and stellar contamination

ROBERT implements the wavelength-dependent transit light source effect (TSLE)
as a typed transform applied to a planet-only transmission spectrum. The
POSEIDON one- and two-heterogeneity implementation is the primary numerical
reference. This feature models unocculted, spatially uniform spectral
components; it is not a stellar-surface map or a spot-crossing light-curve
model.

## Definitions and equation

Let \(I_{\rm phot}(\lambda)\) be the specific intensity (ROBERT uses
disk-averaged surface radiance in `W m^-3 sr^-1`) of the immaculate
photosphere. Active region \(i\) has projected visible-disk covering fraction
\(f_i\) and spectrum \(I_i(\lambda)\). ROBERT defines the disk-integrated
out-of-transit radiance as

\[
I_{\rm disk}(\lambda) =
\left(1 - \sum_i f_i\right) I_{\rm phot}(\lambda)
+ \sum_i f_i I_i(\lambda),
\]

where every \(f_i\) lies in \([0, 1]\) and \(\sum_i f_i \le 1\). Fractions are
projected fractions of the visible stellar disk, not fractions of the full
spherical surface.

Let \(I_{\rm chord}(\lambda)\) be the spectrum occulted by the planet. The
uncontaminated planetary transit depth \(D_{\rm planet}(\lambda)\),
multiplicative contamination factor \(\epsilon(\lambda)\), and observed depth
are

\[
\epsilon(\lambda) =
\frac{I_{\rm chord}(\lambda)}{I_{\rm disk}(\lambda)}, \qquad
D_{\rm obs}(\lambda) =
\epsilon(\lambda) D_{\rm planet}(\lambda).
\]

The POSEIDON/Rackham subset assumes an immaculate transit chord,
\(I_{\rm chord}=I_{\rm phot}\), giving

\[
\epsilon(\lambda) =
\left[1 - \sum_i f_i
\left(1 - \frac{I_i(\lambda)}{I_{\rm phot}(\lambda)}\right)\right]^{-1}.
\]

ROBERT preserves POSEIDON's float64 operation order for this subset. A cool
unocculted spot generally has \(I_i/I_{\rm phot}<1\), so \(\epsilon>1\) and
the observed transit is deeper. A hot facula generally gives \(\epsilon<1\).
Mixed spot and facular terms add inside the denominator; contamination factors
must not be multiplied as independent corrections.

## Stellar spectra and preparation

The pure `StellarContaminationModel` accepts already prepared synthetic or
physical spectra, which keeps algebraic tests independent of external data.
`prepare_stellar_contamination_model` reuses ROBERT's existing stellar
spectrum path for the photosphere, every active region, and an optional chord:

- `phoenix` uses the STScI Synphot PHOENIX catalog selected by temperature,
  `log_g_cgs`, and metallicity. `PYSYN_CDBS` must point to the reference-data
  root above `grid/phoenix`. Preparation rejects spectral extrapolation and
  non-positive flux and occurs before likelihood evaluation.
- `blackbody` is an explicit controlled approximation for examples and
  algebraic sensitivity tests. It is not ROBERT's stellar-atmosphere
  validation standard.

Spot temperatures must be lower than the configured photosphere temperature;
facular temperatures must be higher. A generic `heterogeneity` may lie on
either side. Region log gravity and metallicity default to the photosphere
values. Current YAML retrieves covering fractions while temperatures and
stellar-grid coordinates are fixed at model-construction time. This avoids
stellar file access in the likelihood hot path. Temperature retrieval will
require a separately validated, precomputed interpolation grid.

## Resolution and instrument order

ROBERT applies \(\epsilon D_{\rm planet}\) to the native transmission result
inside `ParameterizedTransmissionForwardModel`, before an external instrument
response or top-hat binning operation. Multiplying separately binned
\(D_{\rm planet}\) and \(\epsilon\) is not equivalent when both vary within a
bin; a regression test enforces the native-first ordering.

Configured correlated-k retrievals prepare mode-specific stellar radiances on
the same band grid as each mode-specific opacity/RT calculation. PHOENIX bin
edges trigger flux-conserving radiance averages. This is consistent with
ROBERT's existing band-model contract, but it is not a substitute for a
higher-resolution line-by-line stellar/planet cross-term calculation when
sub-bin covariance is scientifically important.

The RT result's effective radii and annulus-area contributions remain
planet-only diagnostics. Only `transit_depth` is contaminated; metadata states
this convention explicitly.

## POSEIDON and Rackham scope comparison

POSEIDON 1.4 provides `one_spot` and `two_spots` (spot plus facula) models and
applies the Rackham factor to its computed native transmission spectrum before
later instrument handling. Its retrieval path precomputes stellar spectra over
temperature/log-g grids. The benchmarked POSEIDON model assumes an immaculate
transit chord and does not expose a distinct chord spectrum.

ROBERT deliberately differs in four small ways:

1. The spectrum-based core supports any number of named regions, while strict
   YAML documents the maintained spot/facula use cases.
2. Mixture closure and positive finite spectra are validated explicitly.
   POSEIDON's arithmetic functions themselves do not enforce these checks.
3. ROBERT allows \(\sum_i f_i=1\) when the resulting disk radiance remains
   positive; POSEIDON's tutorial describes the remaining photosphere fraction
   using a strict less-than-one convention.
4. An explicit `transit_chord_temperature_k` or Python-API chord spectrum is a
   generalized flux-ratio extension. It has analytic tests but is outside the
   current POSEIDON parity claim.

The underlying formalism follows [Rackham, Apai & Giampapa (2018), ApJ 853,
122](https://doi.org/10.3847/1538-4357/aaa08c). The implemented reference is
[POSEIDON v1.4.0 `stellar.py` at commit
`d163221`](https://github.com/MartianColonist/POSEIDON/blob/d1632214c9f3087e367a8d752454f3668fc30e18/POSEIDON/stellar.py),
supplemented by the official [stellar-contamination
tutorial](https://poseidon-retrievals.readthedocs.io/en/latest/content/notebooks/transmission_stellar_contamination.html)
and [POSEIDON code paper](https://arxiv.org/abs/2410.18181).

## Degeneracies and limitations

TSLE amplitude and slope can be degenerate with reference planet radius,
atmospheric scale height, haze/Rayleigh slopes, molecular bands shared by cool
stellar spectra and planetary opacity, dataset offsets, and stellar-grid
systematics. Priors on coverage and temperature therefore carry physical
information and should not be interpreted as neutral nuisance choices.

The model does not include limb darkening, center-to-limb spectral variation,
occulted spot-crossing anomalies, time-variable active regions, chromospheres,
flares, arbitrary surface maps, or joint transit-light-curve inference.
Coverage fractions describe one observation epoch and one disk-mixture
convention.

## Validation boundary

`examples/benchmark_poseidon_stellar_contamination.py` executes the unmodified
official POSEIDON 1.4 function bodies from a commit-verified source checkout.
Homogeneous, cool-spot, hot-facula, and mixed cases span 0.6--12 micron and
multiply a nonconstant synthetic planetary spectrum. The compact oracle is
`tests/fixtures/poseidon_stellar_contamination_v1_4_0.json`; full generated
products remain under ignored `examples/outputs/`.

The accepted benchmark has zero float64 residual in contamination factor and
observed depth, within predeclared absolute tolerances of `2e-15` in factor,
`2e-17` in depth, and `5e-18` RMS depth. This validates the implemented
stellar-contamination transform in those tested cases. It does **not** validate
PHOENIX or other stellar atmosphere grids, active-region evolution, the
explicit-chord extension, occulted spot crossings, or arbitrary stellar
surface maps.
