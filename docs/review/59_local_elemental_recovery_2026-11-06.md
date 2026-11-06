# Local fixes and elemental recovery — 6 November 2026

## Code changes

The Virga test helper now sets `gravity_model` only for transmission. Emission
fixtures no longer fail before their cloud assertions. The production schema
continues to reject an explicit transmission-only option in emission.

The experimental Python joint-geometry API now exposes the underlying
likelihood to the manifest through an explicit wrapper contract. Its manifest
records likelihood type, known scientific settings, nuisance roles, and
local/global parameter bindings. A changed likelihood must block checkpoint
resume. Built-in covariance matrices and PSD tolerances are fingerprinted,
HRS velocity-parameter roles are retained, and prepared observations keep
their data identity. Custom covariance operators fail safely rather than
using an incomplete identity. The standard YAML runner still selects one
geometry.

Run creation also preserves the explicit null legacy metallicity/C/O
parameter names used by independent FastChem elemental configurations. The
round-trip test covers both top-level and regional chemistry.

These fixes do not complete joint posterior/profile collection or a physical
joint emission/transmission retrieval. See the
[joint API tutorial](../tutorials/04_joint_emission_transmission.md).

## Recovery protocol

This is one cloud-free, fixed-temperature transmission injection. It tests the
configured independent FastChem elemental parameters with full equilibrium
composition. It is not a fit to WASP-178b observations.

| Setting | Value |
| --- | --- |
| Temperature | Fixed, isothermal, 1800 K |
| Pressure | 80 layers from 100 to 1e-5 bar |
| Data | 48 bins from 2.8 to 5.15 microns |
| Uncertainty | Independent Gaussian, 20 ppm per bin |
| Molecular opacity | Native R=1000 H2O, CO, SiO KTA inputs, all 20 g ordinates |
| Gas combination | Random overlap |
| Geometry | Spherical transmission, inverse-square gravity, reference pressure 1 bar |
| Sampler | One PyMultiNest retrieval, 50 live points, four MPI ranks |
| Thread limit | One numerical thread per rank |
| Evidence tolerance | dlogz = 0.5 |
| Iteration cap | None (`multinest_max_iterations: 0`) |
| Seed | 20261104, retained for comparison with the earlier test |

| Parameter | Injected value | Uniform prior bounds |
| --- | --- | --- |
| log10(n_O/n_H) | -3.18 | [-3.8, -2.8] |
| log10(n_C/n_H) | -3.70 | [-4.3, -3.3] |
| log10(n_Si/n_H) | -4.28 | [-4.9, -3.9] |

Hydrogen, helium, and the unmapped elements retain the input solar abundance
convention. The temperature, radii, gravity inputs, and cloud-free model are
fixed. The test does not measure degeneracy with a fitted temperature profile.

The source tables are the existing shared native-wavelength subsets prepared
on 4 November. Each file is 30,368,240 bytes and has 639 wavelength points,
22 pressure nodes, 27 temperature nodes, and 20 g ordinates. They retain exact
native coefficients, pressure/temperature grids, and quadrature weights.
They are not cross-section HDF files. No opacity file is copied into this run
or committed to Git.

The new injection evaluates the current model at the stated truth and adds
the exact saved noise vector. The noise SHA-256 is
`c22bf6d69bba31361251a242430a517662cba507effc5a7f76077156d0eea5d5`.
The compact result record retains the source hashes and code revision.

The audit initially missed the completed 4 November fixed-temperature
recovery. This new run is a fresh code/regression check with the same noise;
it is not an independent noise realization or a posterior calibration
ensemble. Existing checkpoints and earlier results remain unchanged.

## Interpretation and products

H-minus is configured, but its bound-free contribution is outside this band
and the implemented free-free policy is zero below 2500 K. This test therefore
checks elemental retrieval and the configured full-composition path; it does
not validate a nonzero H-minus signal.

Use the full weighted posterior for elemental interval summaries. Use the
same 100 weighted draws for spectra and atmospheric profiles. Save the draw
indices, parameter vectors, numerical products, and quantile probabilities.
The representative spectrum is the pointwise posterior median, with central
68.2689% and 95.45% intervals. Figures use the WASP-15b style and external
captions.

Full arrays, chains, configurations, driver scripts, and PNG/PDF figures stay
in the isolated external run. Git retains only the compact result record and
this description, in accordance with the [data policy](../data_policy.md).
