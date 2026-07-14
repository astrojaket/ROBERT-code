# NEMESIS-Informed Vertical-Profile Optimal Estimation

## Scope and provenance

This increment adds pressure-resolved optimal estimation (OE) to ROBERT and
uses NEMESIS conventions where they are scientifically useful. It does not
claim full NEMESIS forward-model parity.

The comparison was based on:

- the [official NEMESIS documentation](https://nemesiscode.github.io/) and
  [manual](https://nemesiscode.github.io/pdf/Nemesis_3.0.pdf);
- the
  [archNEMESIS model-parameterisation documentation](https://archnemesis.readthedocs.io/en/latest/documentation/model_parameterisations.html)
  and
  [retrieval documentation](https://archnemesis.readthedocs.io/en/latest/documentation/retrievals.html);
- [`nemesiscode/radtrancode`](https://github.com/nemesiscode/radtrancode) commit
  `4aa5395aecbfc032426b5f154392842db275765b`;
- [`archnemesis/archnemesis-dist`](https://github.com/archnemesis/archnemesis-dist)
  commit
  `d4dd8cb2eb07479291c30e929580e83aaf9155c0`; and
- the official `patrickirwinoxford/docker_nemesis:latest` image, digest
  `sha256:26332a3cecb757eca8f1d62fa5b6e8dc46b25b74515dd05be77ed9b2bf13f486`.

The legacy Jupiter CIRS example was run inside that Docker image through three
reported retrieval iterations. Its accepted steps reduced the objective by
86.1% and then 81.1%. This is a useful executable smoke test, but it is not the
HAT-P-32b parity result described below.

## What NEMESIS does

### Continuous and sparse profiles

NEMESIS model 0 retrieves a value at every atmospheric profile level. Its
apriori file supplies the profile, uncertainty, and a correlation length. Model
25 instead retrieves a smaller number of pressure nodes and interpolates onto
the calculation grid. These are complementary choices: model 0 is appropriate
when averaging kernels will be used to report the actual resolved information;
model 25 reduces dimension and forward-model calls when the data cannot support
many independent levels.

Temperature is represented linearly. Positive quantities such as gas VMR and
aerosol abundance are represented in natural-log coordinates. For pressure
levels `p_i` and `p_j`, NEMESIS constructs the prior covariance as

```text
Sa[i,j] = sigma[i] sigma[j]
          exp(-abs(ln(p_i / p_j)) / correlation_length)
```

The correlation length is therefore expressed in pressure scale heights. A
100-element profile is not a claim of 100 independent measurements: the trace
of the averaging kernel is the degrees of freedom for signal (DOFS).

### Gases

NEMESIS can retrieve one scaling factor, a compact profile, or a continuous VMR
profile. ROBERT now supports the continuous case with one explicitly named
log-VMR state value per pressure node. Multiple species are independent state
blocks by default; any cross-species covariance must be supplied deliberately.
The new `SplineFreeChemistry` maps pressure-node parameters onto the atmospheric
grid and retains the existing H2/He background-fill and composition checks.

### Clouds and surfaces

The important NEMESIS cloud choices are not all layer-by-layer:

- model 0 retrieves a continuous aerosol profile;
- model 9 uses cloud-base altitude, optical depth, and fractional scale height;
- model 32 uses cloud-base pressure, integrated aerosol amount/optical depth,
  and fractional scale height, with a rapid decline below the base;
- model 444 retrieves particle size together with the wavelength-dependent
  imaginary refractive index; and
- separate machinery represents fractional/broken cloud cover and surface
  boundary properties.

ROBERT's state-vector layer now represents positive aerosol and cloud-fraction
profiles with the same log transform and pressure correlation used for gases.
That is state-vector support, not yet a complete connection to every cloudy
radiative-transfer backend. The next cloud increment should connect continuous
aerosol number/optical-depth profiles to the existing Mie optical properties,
then add compact base-pressure/scale-height models and surface emissivity or
albedo profiles. Those additions need cloudy and rocky-atmosphere golden tests,
not just transform unit tests.

### Iteration and diagnostics

NEMESIS uses an adaptive Marquardt brake. A successful step reduces the brake
factor; a rejected step increases it and retries a smaller move. ROBERT now
implements this accepted/rejected-step behavior and returns:

- the posterior covariance;
- gain and averaging-kernel matrices;
- measurement-error and smoothing-error covariance;
- the final finite-difference Jacobian; and
- DOFS as `trace(averaging_kernel)`.

A fixed full measurement covariance and optional forward-model error may be
included. This matters for high-S/N JWST data: treating model inadequacy as
zero forces OE to interpret systematic residuals as vertical information.

## HAT-P-32b injection/recovery

`examples/benchmark_hat_p_32b_vertical_oe.py` is a reproducible cloud-free
experiment using the bundled HAT-P-32b median temperature and six-gas VMR
profiles. It:

1. constructs 100 log-spaced wavelength bins from 2.9 to 5.15 microns;
2. recompresses the original HAT-P-32b correlated-k tables onto those bins;
3. creates a zero-noise eclipse spectrum with 30 ppm independent errors;
4. retrieves temperature at all 100 pressure levels using a 1.5-scale-height
   covariance and a deliberately displaced prior; and
5. saves the Jacobian, averaging kernel, error covariances, spectra, and
   NEMESIS interchange tables.

The first complete 100-bin/100-layer run converged in seven iterations. The
temperature RMSE improved from 188.8 K in the prior to 56.1 K in the retrieved
profile. The DOFS was 5.94. The scientifically relevant result is therefore
roughly six independently sounded temperature combinations, not 100 resolved
temperatures. Reporting the fine grid remains useful because the averaging
kernels show where and how those combinations are localized.

The initial opacity preparation and compiled-kernel call took about 3.06 s.
After compilation, repeated forward calls took a median 0.060 s (about 16.7
calls/s). The seven-iteration forward-difference OE took 50.2 s on the
development machine. Central differences remain available for validation but
roughly double the Jacobian work.

Run it in the project environment with:

```bash
conda run -n robert-exoplanets python \
  examples/benchmark_hat_p_32b_vertical_oe.py \
  --output-dir /tmp/robert-hat-p-32b-100x100-oe
```

The original `.kta` directory can be overridden with `--kta-dir`. The bundled
ROBERT opacity archives have a fixed spectral grid, so they are intentionally
not silently treated as exact 100-bin inputs.

## NEMESIS comparison contract and remaining work

The benchmark writes `hat_p_32b_atmosphere.txt`,
`hat_p_32b_spectrum_100_bins.txt`, and `nemesis_comparison_contract.json`.
They fix pressure, temperature, VMR, bin edges, uncertainty, radii, gravity,
stellar temperature, geometry, and observable conventions for a second-code
run.

A scientifically valid like-for-like NEMESIS result still requires:

1. NEMESIS-readable k tables from the same line lists, pressure/temperature
   grid, quadrature, and wavelength bins, with checksums recorded;
2. identical CIA, random-overlap, hydrostatic, lower-boundary, and disc
   integration choices;
3. an identical stellar spectrum and planet/star normalization; and
4. a completed Docker retrieval whose `.mre`, averaging kernel, and covariance
   are archived with the executable/image provenance.

Until those conditions are met, a ROBERT-versus-NEMESIS spectral residual is a
whole-model comparison, not a diagnostic of ROBERT's OE solver. The current
Docker result establishes that the legacy executable runs; the interchange
contract makes the HAT-P-32b comparison concrete and auditable rather than
overstating parity.

## Recommended validation sequence

1. Sweep correlation length and prior amplitude, reporting DOFS and
   prior-sensitivity rather than a single best profile.
2. Retrieve temperature plus one continuous VMR profile, then add gases one at
   a time to expose cross-talk through off-diagonal averaging kernels.
3. Repeat at realistic JWST covariance and with a forward-model-error floor.
4. Add compact NEMESIS-style and continuous aerosol profile cases.
5. Add grey and spectral surface emissivity/albedo for temperate rocky planets.
6. Validate each case against the Docker-pinned NEMESIS executable before using
   hybrid nested-sampling-to-OE profile refinement for science interpretation.
