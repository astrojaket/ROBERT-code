# Tutorial 4: Join emission and transmission retrieval components

This tutorial shows how to connect separate emission and transmission models
to one retrieval parameter set. The joint geometry API is a Python prototype.
It connects forward models, observations, likelihoods, and parameters. It does
not provide new atmospheric physics or a production workflow.

Use a component only when its observing geometry and data exist. A
nontransiting planet or brown dwarf has no transmission geometry. Use its
emission or high resolution spectrum alone. Do not add a blank transmission
component to make a joint fit.

The current component API is in
[robert_exoplanets.retrieval.joint_geometry](../../src/robert_exoplanets/retrieval/joint_geometry.py).
It returns a HeterogeneousRetrievalProblem. The API tests check parameter
bindings and likelihood composition. They do not validate the science of a
target model. External GJ 357 b pilots compare synthetic joint and separate
fits, including an airless control and thermal mismatch. Their blackbody
star, approximate detector bins and unspecified CO2 broadener limit their
science use. Calibrated recovery ensembles, theory grid validation, and
production postprocessing remain future work.

The standard posterior collector and post-processing path do not yet handle
joint component state as one result. They do not provide joint-aware products
for component-specific offsets, atmosphere profiles, or saved prediction
artifacts. A sampler-compatible problem object does not prove that these
posterior products are complete. Keep the joint API as a composition
prototype until result collection, profile summaries, and figure generation
support each component and pass recovery checks.

## 1. Keep the three science stages separate

| Stage | Test | Limit on the claim |
| --- | --- | --- |
| UNITY | Compare a small set of physical alternatives for the nine selected planets. Apply the same instrument response to each prediction and test it against the expected precision. | Predictions are simulated values. Label them as predictions, not observations. Test confounding effects and withheld targets before reporting a population relation. |
| Cooler worlds | Choose one CO/CH4 or cloud quantity. Compare it at stated pressures and local temperatures. Keep gravity and heating source distinct. | Do not put a planet's equilibrium temperature and a brown dwarf's effective temperature on one axis as though they were the same measure. Use a published benchmark as prior work only. |
| Rocky worlds | Fit each measured geometry with its own forward model and likelihood. Test an atmospheric case, an airless control, and a mismatch case. | Compare absolute fit as well as relative evidence. Allow the result that neither tested model fits. Theory predictions do not count as atmospheric detections. |

For UNITY, use the published
[Roth, Parmentier and Hammond (2024) climate grid](https://doi.org/10.1093/mnras/stae984)
as the initial physical comparison. Preserve the model identity and check
target coverage in irradiation, gravity and rotation. Apply the observation
response to the published spectra before assessing residuals. An available
climate model is not automatically a fit to the target. Account for the
grid's fixed radius, interior heating and chemistry assumptions when mapping
it to a target. Use the
[versioned data archive](https://doi.org/10.5281/zenodo.10960010).
UNITY is a forecast at this stage. Display its spectra only over continuous
NIRSpec G395M coverage, without a detector gap. For BOTS SUB2048, the
[approximate coverage is 2.87–5.18 μm](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-operations/nirspec-bots-operations/nirspec-bots-wavelength-ranges-and-gaps).
Keep the target response and bin edges explicit; a plot range is not an
instrument response. WASP-178b supplies an existing ERC method example and
the bridge to resolved ground-based lines. Its emission-only example does
not establish a joint ground/space result or the nine-target population test.
Model ranking requires the actual data, uncertainty and covariance; a synthetic
forecast only tests whether observations can distinguish those models.
The PDRA will run bespoke GCMs to predict both day and limb spectra for
the selected targets. These models can test physics and target properties
that the initial grid does not cover. Their temperature, chemistry and cloud
fields will inform the spatial structure used in the future retrieval.
Emission will integrate intensity over the visible disk. Transmission will
integrate slant optical depth through the extended terminator, including
day–night gradients and morning/evening differences. The current joint
prototype keeps the standard 1D geometry of each component; it does not
perform these spatial integrations. Keep the forward comparison and the
retrieval tied to the same physical scenario. Do not use the same observations
first to select a narrow prior and then again as an independent likelihood;
compare climate hypotheses within one stated probabilistic model.

The WASP-178b day–limb pilot will compare emission-only, transmission-only
and joint inference using the observed G395M eclipse spectrum and published
[G395H transit spectrum](https://doi.org/10.3847/1538-3881/adc117).
Keep the real G395H detector gap and bin edges in this data comparison.
The UNITY forecast plot rule does not remove a gap from an observed G395H
spectrum. A numerical transit spectrum with uncertainties is required before
running the two fits that use transmission. The lead author's
[public numerical data file](https://github.com/jlothringer/jlothringer.github.io/blob/main/W178_defsb10pix.txt)
provides wavelength centres, bin widths, transit depths and uncertainties.
Record the repository revision, source checksum and wavelength selection
when preparing this input. Separate 1D day and limb profiles
are a pilot approximation. Shared free molecular abundances are a conditional
model and do not account for different local dissociation. The future model
will share elemental composition and compute local chemistry from the
GCM-informed atmospheric state.

For a resolved line comparison, apply the same filtering and instrument
response to data and model. A continuum-normalised spectrum does not measure
absolute flux. Keep epochs and calibration terms in the observation model.

## 2. Build the joint problem

The component binding map connects each forward model's local parameter names
to names in the global prior set. A repeated global name ties those local
values. Different global names keep them independent. Use the exact local
names that each model declares in required_parameters.

The example below gives each geometry its own local temperature and gas
parameter. The metallicity and C/O values are shared. This can be suitable
when each geometry sees a different temperature profile but the same bulk
elemental composition. It does not say that all retrieved gas VMR profiles
must be equal. A local chemistry model may derive those profiles from local
temperature and pressure.

~~~python
from collections.abc import Mapping

from robert_exoplanets.likelihoods import GaussianLikelihood
from robert_exoplanets.retrieval import RetrievalParameterSet
from robert_exoplanets.retrieval.joint_geometry import (
    JointGeometryComponent,
    build_joint_geometry_problem,
)


def make_joint_problem(
    *,
    transmission_forward_model,
    transmission_observation,
    emission_forward_model,
    emission_observation,
    transmission_bindings: Mapping[str, str],
    emission_bindings: Mapping[str, str],
    parameters: RetrievalParameterSet,
    prior_scenario: str,
    theory_source: str,
):
    transmission = JointGeometryComponent(
        name="transmission",
        forward_model=transmission_forward_model,
        likelihood=GaussianLikelihood(
            offset_parameter=None,
            jitter_parameter=None,
        ),
        observation=transmission_observation,
        required_parameters=transmission_forward_model.required_parameters,
        parameter_bindings=transmission_bindings,
        metadata={"role": "transmission"},
    )
    emission = JointGeometryComponent(
        name="emission",
        forward_model=emission_forward_model,
        likelihood=GaussianLikelihood(
            offset_parameter=None,
            jitter_parameter=None,
        ),
        observation=emission_observation,
        required_parameters=emission_forward_model.required_parameters,
        parameter_bindings=emission_bindings,
        metadata={"role": "emission"},
    )
    return build_joint_geometry_problem(
        name="target-joint-emission-transmission",
        components=(transmission, emission),
        parameters=parameters,
        metadata={
            "prior_scenario": prior_scenario,
            "theory_source": theory_source,
        },
    )
~~~

Supply the observations, forward models, bindings, and a
RetrievalParameterSet from a versioned run input. The local binding keys
must match the model parameter names. For example, map each model's local
temperature to a separate global temperature, and map each model's local
metallicity and C/O names to shared global names. Check that every global name
is used and every local name is bound.

Call the function with a broad baseline parameter set. Call it again with a
theory-informed parameter set. Keep the data, forward models, likelihoods,
response, and parameter bindings the same when the only change under test is
the prior. For an emission-only or transmission-only fit, pass only that
component and a matching parameter set. The builder rejects unused global
parameters.

## 3. Treat priors as part of the model

The broad baseline and theory-informed run answer different conditional
questions. Each prior must be a normalized probability distribution. Evidence
is conditional on that distribution and its support. Record the prior source,
scenario, version, and numerical limits in the run metadata.

The current RetrievalParameterSet holds one prior for each named global
parameter. A rectangular set of independent priors cannot represent a
correlated GCM constraint. Do not clip points inside the likelihood to imitate
a correlated theory distribution. The evidence would still use the broader
prior normalization.

A correlated theory grid needs a validated, normalized joint prior transform
or another sampler input that preserves the grid's joint probability. Do not
invent numerical theory limits. Derive them from a cited, versioned grid and
record how its weights and uncertainty enter the prior. Test the transform
before comparing evidence.

Use chemistry parameters that match the physical question. Temperature can
differ between a transmission limb and an emission day side. A chemistry
model can use those local profiles with shared metallicity and C/O. Do not
tie every gas VMR across both geometries unless the model justifies that
assumption. A free VMR fit is a separate assumption from a shared elemental
composition.

## 4. Keep physical inputs and data terms clear

Use the following limits when a theory grid or target likelihood is added:

- Use measured mass and radius in a documented mass/radius likelihood when
  the model needs them. Do not treat density as certain from theory. If density
  is derived from the same mass and radius, do not add it as an independent
  second constraint unless the joint covariance accounts for the reused data.
- Atmospheric escape depends on system history and replenishment. A static
  present-day grid does not prove atmospheric survival. Future PDRA models
  must state the stellar history, escape prescription, and replenishment
  assumptions.
- Albedo and clouds change the energy balance and temperature profile. Their
  effect needs a radiative-convective equilibrium model. Independent albedo
  and cloud axes without that energy calculation are a sensitivity study,
  not a self-consistent prediction.
- Future GCM inputs should record the target, gravity, irradiation, pressure
  grid, chemistry, cloud settings, model version, and output weights. Use
  them to define a normalized theory prior only after the grid is checked.

For rocky planets, retrieve local day and limb thermal parameters before
imposing a climate relation. Report any temperature contrast at a stated
pressure. Transmission and emission can sample different pressure ranges;
this contrast alone does not give a transported-power fraction. The limb
is not the nightside. An energy-balanced circulation model must connect
the regional profiles, surface, albedo and clouds to heat transport.
[Wordsworth (2015)](https://doi.org/10.1088/0004-637X/806/2/180) and
[Koll (2022)](https://doi.org/10.3847/1538-4357/ac3b48) provide rocky-planet
transport and stability models. Use their stated domains rather than a
universal pressure threshold. Test composition, pressure and redistribution
degeneracies, as found for 55 Cancri e by
[Zilinskas et al. (2025)](https://doi.org/10.1051/0004-6361/202554062).

The current builder adds each component log likelihood. It therefore assumes
that separate components have no cross-covariance. Keep detector and
instrument observations on their own wavelength grids. Use
GaussianLikelihood for independent errors. Use
CorrelatedGaussianLikelihood for a supported covariance within one
component. A covariance between components needs one explicit joint
observation and likelihood, or a new validated likelihood API. Do not replace
known covariance with diagonal errors.

Map likelihood nuisance parameters in the same way as model parameters. For
example, an additive offset uses likelihood_parameters=("offset",) and a
binding from local offset to a global parameter. A jitter term uses the
observation's flux unit and adds in quadrature to its uncertainty. Give every
nuisance parameter a source and a normalized prior. Do not fit a free offset
when calibration does not support it.

For any linear data filter F, propagate covariance as C_out = F C_in F.T.
Preserve the retained residual rank. The public WASP-77Ab products do not
provide the cross-order or cross-detector covariance used by the original
papers. The current comparison states its independent block assumption in
its [validation protocol](../review/52_wasp77ab_real_joint_retrieval.md).

## 5. Current benchmark: prior work only

The supported local benchmark is WASP-77Ab. It has the 2020-12-14 IGRINS
pre-eclipse cube and Smith et al. model templates from the
[Zenodo record](https://doi.org/10.5281/zenodo.10382053). The separate August
et al. NIRSpec table has 150 bins, split between NRS1 and NRS2. See the
[source protocol](../review/52_wasp77ab_real_joint_retrieval.md),
[Smith et al. (2024)](https://doi.org/10.3847/1538-3881/ad17bf), and
[August et al. (2023)](https://doi.org/10.3847/2041-8213/ace828).

This benchmark is a hot-Jupiter method check, not a cooler-world result. The
high resolution LBL check uses the real IGRINS pixel grid for 19 K-band
orders. It compares sampled model spectra after rotation, the Gaussian
R=45,000 line-spread response, and pixel integration. It does not fit the
cube flux. The selected LBL strides were 2 for HRS and 25 for NIRSpec. HRS
stride 2 had RMS and maximum fractional differences of 4.09e-5 and
3.19e-4 against stride 1. The run took 393.64 seconds, used one CPU, and
reached 1,830,780,928 bytes of peak RSS. All numerical thread settings were
one. This sampling check does not change the native stride-1 requirement.
The synthetic retrieval gate rejects stride 2. Use stride 1 for retrievals
and synthetic injection-recovery tests.

The NIRSpec check compares the published data with the Smith R=500,000
median model after top-hat integration over each detector's published bins.
It passes at chi-square per point 0.72475 for 150 points. The reported
runtime was 11.29 seconds and peak RSS was 208,814,080 bytes. This is a fixed
published model comparison, not an independent atmosphere retrieval. The
figure shows the two detector segments separately and keeps their 0.119
micron gap.

The task-local records and figure files are outside Git:

- [LBL convergence report](/Users/jaketaylor/Developer/ROBERT-runs/tracy-erc-20261005/cooler_benchmark/wasp77ab_lbl_sampling.json)
- [NIRSpec validation report](/Users/jaketaylor/Developer/ROBERT-runs/tracy-erc-20261005/cooler_benchmark/wasp77ab_lrs_validation.json)
- [NIRSpec paper panel, PNG](/Users/jaketaylor/Developer/ROBERT-runs/tracy-erc-20261005/cooler_benchmark/wasp77ab_lrs_paper_panel.png)
- [NIRSpec paper panel, PDF](/Users/jaketaylor/Developer/ROBERT-runs/tracy-erc-20261005/cooler_benchmark/wasp77ab_lrs_paper_panel.pdf)
- [NIRSpec paper panel, SVG](/Users/jaketaylor/Developer/ROBERT-runs/tracy-erc-20261005/cooler_benchmark/wasp77ab_lrs_paper_panel.svg)

The reports record input checksums and complete run metrics. The figure uses
the published asymmetric errors. The validation statistic uses their mean
absolute error, as declared in the report.

To remake the paper-style panel from these verified inputs, run the external
plot helper. It writes PNG, PDF, and SVG files beside the helper:

~~~bash
conda run -n robert-exoplanets env \
  PYTHONPATH=/Users/jaketaylor/Developer/ROBERT-code \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  python /Users/jaketaylor/Developer/ROBERT-runs/tracy-erc-20261005/cooler_benchmark/plot_wasp77ab_lrs_paper_panel.py
~~~

To repeat the LBL check from the repository root, set the thread limits and
write the report outside the checkout:

~~~bash
conda run -n robert-exoplanets env \
  PYTHONPATH=/Users/jaketaylor/Developer/ROBERT-code \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  python examples/benchmark_wasp77ab_lbl_sampling.py \
  --report /Users/jaketaylor/Developer/ROBERT-runs/tracy-erc-20261005/cooler_benchmark/wasp77ab_lbl_sampling.json
~~~

To repeat the NIRSpec comparison and write its report and plot:

~~~bash
conda run -n robert-exoplanets env \
  PYTHONPATH=/Users/jaketaylor/Developer/ROBERT-code \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  python examples/validate_wasp77ab_smith2024_lrs.py \
  --data-root /Users/jaketaylor/Developer/ROBERT-code/data/jwst_emission_spectra \
  --template /Users/jaketaylor/Developer/ROBERT-code/external_data/wasp77ab_smith2024/w77_pre_nirspec_best_fit_R500K_scaled.txt \
  --report /Users/jaketaylor/Developer/ROBERT-runs/tracy-erc-20261005/cooler_benchmark/wasp77ab_lrs_validation.json \
  --plot /Users/jaketaylor/Developer/ROBERT-runs/tracy-erc-20261005/cooler_benchmark/wasp77ab_lrs_validation.png \
  --max-memory-gib 1.9
~~~

The HRS and LRS checks do not establish science-grade posterior recovery. The
full real-data shared-atmosphere PyMultiNest run remains deferred to a
cluster. The current HRS/LRS protocol also records that a non-isothermal
temperature model and published covariance definitions are still needed
before a literature reproduction claim.

No HD 4747 B data loader, verified input set, model, or processing record was
found in this checkout or the related project data checked for this task.
Treat that name as a literature fallback until its public data, model, and
processing steps have a verified local record.
