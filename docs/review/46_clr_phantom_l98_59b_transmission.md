# CLR, phantom chemistry, and L 98-59 b transmission validation

## Scope and starting point

The implementation began from local `main` at commit
`92f96a9f126207cab8feb913c93503d959835f25`, exactly even with `origin/main`
after a fetch (zero commits ahead or behind). The pre-change focused
transmission suite passed with 58 tests and one optional native-MultiNest skip.

This work adds a geometry-independent composition prior and MMW treatment. It
is scientifically exercised here only in transmission. Emission construction
and evaluation have regression coverage, but no emission CLR retrieval is
claimed as validated.

## POSEIDON CLR parity

ROBERT implements the `N`-coordinate, `N + 1`-category CLR transform used by
`POSEIDON.retrieval.CLR_Prior`. The omitted category is recovered from unit-sum
closure by free chemistry, so no molecule is structurally designated as the
bulk gas.

The executable comparison in `examples/compare_poseidon_clr_prior.py` extracts
the reference function from the POSEIDON source AST. This avoids importing
POSEIDON's optional MPI, sampler, and opacity stack. The validation is pinned
to:

- POSEIDON commit: `d1632214c9f3087e367a8d752454f3668fc30e18`
- `POSEIDON/retrieval.py` SHA-256:
  `04d4df30bff5a98af5c6bd647a16fce6c401e74c0cf837a0bcdb78ed29200f11`
- CLR lower limit: `log10(VMR) = -12`
- 10,000 samples each for 1, 2, 3, and 6 free coordinates

Across all 40,000 points, the largest absolute difference in a free `log10`
VMR was `1.7764e-15`. Frozen POSEIDON vectors are also part of the normal
pytest suite. The generated record is
The generated comparison JSON is excluded from Git under the repository data
policy and will be archived with the benchmark release on Zenodo.

## Phantom molecule

The optional phantom species is the sole background closure category. It:

- has no line-opacity table;
- contributes zero H2/He Rayleigh scattering and no CIA pair;
- has a positive retrieved molecular mass in amu;
- contributes `X_phantom * mass_phantom` to the composition-weighted MMW.

The same `AtmosphereBuilder`, parameter discovery, opacity-free metadata, and
MMW evaluation are used by emission and transmission factories. A real
parameterized emission-forward evaluation with an opacity-free phantom and
fitted mass is covered by tests.

## Published L 98-59 b data

Both combined spectra were downloaded from Zenodo record
`10.5281/zenodo.14676143` (concept DOI `10.5281/zenodo.14676142`) and their
published checksums verified. The retrieval uses the 218-point, 0.01 micron
Eureka! reduction. Its loader converts ppm to dimensionless transit depth and
splits the spectrum at the native detector gap:

- NRS1: 84 points, 2.875--3.705 microns;
- NRS2: 134 points, 3.835--5.165 microns.

The split permits a fitted NRS2 relative offset.

## Requested ROBERT retrieval

Configuration:
`configurations/l98_59b_clr_transmission_multinest.yaml`.

The requested four gases are represented symmetrically as three free CLR
coordinates (`SO2`, `H2S`, and `CO2`) plus derived `H2`. The run also fits an
isothermal temperature, 1-bar reference-radius scale, and NRS2 offset. It used
real R1000 ExoMol correlated-k tables, 50 live points, 3 local OpenMPI ranks,
and native MultiNest 3.10 through PyMultiNest 2.12.

The run converged after 1,857 likelihood evaluations and 552 replacements.
ROBERT's stored importance-nested-sampling evidence is
`ln Z = 1893.950 +/- 0.050`. The best fit has:

- chi-squared: `188.30` for 212 degrees of freedom;
- reduced chi-squared: `0.888`;
- isothermal temperature: `587 (+134/-250) K` (16th/50th/84th percentiles);
- `log10(X_SO2) = -0.013 (+0.013/-0.868)`;
- NRS2 offset: `-6.1 (+7.9/-6.6) ppm`.

Derived 95% constraints are:

- `X_H2 < 24.11%`;
- `X_CO2 < 80.72%`;
- MMW `> 33.64 amu`.

For comparison, the paper's 2,000-live-point, 11-parameter POSEIDON Figure 8
analysis reports `X_H2 < 24%`, `X_CO2 < 84%`, and MMW `> 20.1 amu` at 2 sigma,
with an SO2-rich solution. The ROBERT temperature also agrees with the paper's
approximately 596 K isothermal result.

This is a strong qualitative and constraint-level comparison, but the evidence
values must not be compared directly. The ROBERT run follows the requested
four-gas simplification, fixes planet mass and the 1-bar lower boundary, uses
R1000 opacity, and has 50 rather than 2,000 live points. POSEIDON additionally
retrieved N2, H2O, CH4, surface pressure, and planet mass and evaluated spectra
at R=20,000.

Local run products, including fit residuals, posterior plots, chains, and the
derived Figure 8 comparison, are under
`examples/outputs/l98_59b_clr_transmission/` and intentionally remain ignored
run artifacts.
