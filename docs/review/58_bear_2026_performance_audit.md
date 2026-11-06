# BeAR 2026 performance audit and ROBERT experiment plan

Date: 2026-11-06. Status: source and paper review; no new performance code or
retrieval was run. The proposals below are not measured ROBERT speed gains.

## Sources and reported timing

The review covers [Wang et al. (2026), arXiv:2611.08489v1](https://arxiv.org/html/2611.08489v1)
and BeAR revision
[`060087edb3eae403fba3165c49476aeb8bc2d947`](https://github.com/NewStrangeWorlds/BeAR/tree/060087edb3eae403fba3165c49476aeb8bc2d947).
The source was obtained over SSH and inspected without building it. The ROBERT
comparison uses `17535f0` and the local changes identified in the
[repository audit](57_repository_audit_2026-11-06.md).

The paper's Table 1 reports a 3.9 ms GPU likelihood, including its forward
model, against 680 ms on 16 CPU cores for 186,552 spectral points on an RTX
5080 desktop. Its 17-parameter MultiNest run took 42 minutes; eight RTX 4090s
took about 14 minutes. Forward throughput scales better than total retrieval
time because the sampler becomes limiting. Injection tests found abundance-ratio
uncertainties too narrow by about a factor of two. These are reported BeAR
results, not matched ROBERT benchmarks. [Source: Sections 4 and Appendix C](https://arxiv.org/html/2611.08489v1).

## What the source establishes

### Keep the repeated calculation on the device

BeAR's GPU retrieval calls the GPU forward model, then the low- and
high-resolution device likelihood paths. Spectrum and observation buffers
persist between calls. The reduction returns scalar double likelihood values
to the host; diagnostic spectrum copies use a separate path. The joint route
can make separate scalar transfers for its likelihood branches.

Evidence: [`retrieval.cpp`, GPU retrieval call](https://github.com/NewStrangeWorlds/BeAR/blob/060087edb3eae403fba3165c49476aeb8bc2d947/src/retrieval/retrieval.cpp#L392),
[`log_like_kernels.cu`, scalar reduction/copies](https://github.com/NewStrangeWorlds/BeAR/blob/060087edb3eae403fba3165c49476aeb8bc2d947/src/CUDA_kernels/log_like_kernels.cu#L60).

ROBERT already has a compiled scalar likelihood contract in
[`metal/problem.py`](../../src/robert_exoplanets/metal/problem.py), persistent
device opacity arrays, and device projections/likelihoods. A new generic GPU
wrapper alone is therefore not the missing upgrade. The useful work is to
extend a complete supported physical path and measure the transfers and
synchronization that remain.

### Fuse opacity interpolation and species accumulation

The batched BeAR kernel groups work by atmospheric grid point. A spectral
thread interpolates each species in that group and accumulates its contribution
in a register. It writes the sum once. Reusable pinned host/device staging
buffers hold the small changing interpolation metadata. This avoids a separate
large interpolated result and device launch for every species.

Evidence: [`batched_opacity_kernels.cu`](https://github.com/NewStrangeWorlds/BeAR/blob/060087edb3eae403fba3165c49476aeb8bc2d947/src/CUDA_kernels/batched_opacity_kernels.cu#L16).

The closest ROBERT candidate is the direct-cross-section path in
[`metal/opacity_sampling.py`](../../src/robert_exoplanets/metal/opacity_sampling.py).
Its interpolation returns a species/layer/wavelength array before optical-depth
assembly sums species. First inspect the compiled graph and peak allocations:
XLA may already fuse some operations. Then test a spectral chunk or fused
accumulation kernel if materialization remains costly.

ROBERT's CPU opacity-sampling provider already fuses interpolation and species
addition in `_mixture_cross_section_kernel` in
[`opacity/opacity_sampling.py`](../../src/robert_exoplanets/opacity/opacity_sampling.py).
That CPU design is not a missing feature.

This design does not replace conservative correlated-k random overlap. Direct
cross sections add at common physical wavelengths; k-distributions require
their declared gas-combination method. Preserve ROBERT's g ordinates and the
CPU reference. The [July BeAR/Exo Skryer review](31_bear_exoskryer_correlated_k_review.md)
already explains this distinction.

### Reuse the HRS filter rather than refit its basis at every proposal

BeAR prepares a per-order temporal projection and the removed data component
once. The trial model is shifted, scaled by that stored data component when
re-injection is selected, and passed through the fixed projection. The source
also prepares data-only variance terms before repeated likelihood calls.

Evidence: [`highres_observation.cpp`, preparation](https://github.com/NewStrangeWorlds/BeAR/blob/060087edb3eae403fba3165c49476aeb8bc2d947/src/observations/highres_observation.cpp#L97),
[`highres_loglike_kernels.cu`, filtering](https://github.com/NewStrangeWorlds/BeAR/blob/060087edb3eae403fba3165c49476aeb8bc2d947/src/CUDA_kernels/highres_loglike_kernels.cu#L771).

ROBERT already supports prepared fixed linear data/model filters in
[`high_resolution_observation.py`](../../src/robert_exoplanets/instruments/high_resolution_observation.py).
Its separate Smith-style time-resolved likelihood deliberately reconstructs
the injected model with a fresh SVD at each proposal. The scalar path already
streams one order at a time; the full cubes are diagnostic products.
See [`_stream_statistics`](../../src/robert_exoplanets/likelihoods/time_resolved_high_resolution.py).

The missing experiment is a prepared fixed *temporal* projection/re-injection
mode for that time-series path. It must have a new method identity. Freezing
the basis changes the likelihood operator; it is not a physics-preserving
backend switch for the current SVD calculation. Masks, per-pixel errors,
removed rank, continuum treatment, and amplitude priors must remain explicit.

### Separate scalar evaluation from diagnostic products

The BeAR source separates `computeHighResModelFlat` from the scalar likelihood.
ROBERT also has spectrum-only thermal integration, a streaming scalar HRS
statistic, and a scalar Metal problem. Preserve those designs. Profile object
construction, array copies, validation, and optional profile extraction before
adding a second layer of acceleration.

One small candidate is already visible: `_stream_statistics` allocates `m2`,
`f2`, `cross`, and `terms`, while the scalar `loglike` caller only sums the
terms. A scalar-only accumulator can avoid those diagnostic arrays without
changing the model-side SVD. Keep the existing statistics API for diagnostics
and require equality with the current scalar reference over saved proposals.

Evidence: [`retrieval.cpp`, diagnostic entry point](https://github.com/NewStrangeWorlds/BeAR/blob/060087edb3eae403fba3165c49476aeb8bc2d947/src/retrieval/retrieval.cpp#L447),
[`ROBERT HRS likelihood`](../../src/robert_exoplanets/likelihoods/time_resolved_high_resolution.py),
[`ROBERT Metal problem`](../../src/robert_exoplanets/metal/problem.py).

### Treat precision and platform as part of the method

BeAR's inspected opacity and high-resolution work arrays use float32, while
some accumulated likelihood values and geometry use double precision. ROBERT
also has float32 experimental device graphs. A broad conversion of production
arrays to float32 would require new error and posterior checks.

The opacity comparison must use a matched direct-cross-section calculation.
The new paper's HRS cloud-free case is not a Virga-cloud comparison. Its
atmosphere and FastChem work remain on the CPU. These scope limits are stated
in [Sections 3.1 and 4](https://arxiv.org/html/2611.08489v1).

CUDA kernels target NVIDIA devices. They do not execute on an Apple GPU.
ROBERT's Metal work is a separate backend with separate limits. FastChem and
Virga costs also need their own profiles: moving the spectral calculation to
a device does not move these external CPU solvers.

### Reuse fixed broadening and check exposure averaging

BeAR's rotational/Gaussian broadening uses reusable device buffers and actual
velocity spacings. ROBERT's WASP-77Ab runner already prepares fixed Gray and
Gaussian operators, but applies them separately through intermediate spectra.
If B0 finds a material response cost, compare a composed or compiled application
with that sequential reference. Check normalization, edges, and line shapes.

Evidence: [`BeAR velocity broadening`](https://github.com/NewStrangeWorlds/BeAR/blob/060087edb3eae403fba3165c49476aeb8bc2d947/src/forward_model/modules/velocity_broadening/velocity_broadening.cpp#L173),
[`ROBERT joint runner`](../../examples/run_wasp77ab_robert_joint_pymultinest.py).

Exposure smearing is a separate scientific extension. BeAR can average over a
per-exposure velocity interval; ROBERT's present time-series schema has no
exposure-duration field and its current model uses one velocity per frame.
Add this only for a target that needs it, then compare with time-resolved
integration. Do not count added physics as a speed improvement.

## Ordered experiments

The expected value below is an engineering assessment from the source. It is
not a measured timing result. Complete the cloud recovery and failure-handling
gates in the repository audit before changing a production run.

| Order | Proposed work | Value and effort | Required result |
| --- | --- | --- | --- |
| B0 | Record stage times for the actual clear, Virga/SH4, and HRS likelihoods using a saved proposal set. | High value, small effort. It identifies the work that matters for each science case. | Separate setup/JIT, atmosphere, chemistry/clouds, opacity/mixing, RT, response/filter, likelihood, transfers, and sampler overhead. Save dimensions, revision, opacity hashes, hardware, threads, memory, and all likelihood values. |
| B1 | Add a fixed temporal-filter/re-injection HRS research mode. | High potential for HRS; medium effort and a scientific operator change. | Compare to an explicit fixed-matrix reference. Test shifted/constant templates, masks, rank removal, normalization, and amplitude signs/priors. Then compare full injection posteriors with the existing SVD mode. |
| B2 | Reduce allocations and Python work in the complete CPU likelihood. | Broad value; small/medium effort. It applies before GPU access is available. | Start with scalar-only HRS accumulation. Reuse static indices/responses and omit diagnostic products on sampler calls. Preserve exact SVD behavior and reference values over saved proposals. Keep public typed objects at setup boundaries. |
| B3 | Test fused, device-resident direct opacity interpolation and accumulation. | Potentially high value on large sampled grids; medium/high effort. | Confirm an allocation or launch bottleneck first. Compare physical samples, interpolation convention, species sum, spectra, and likelihoods against float64 CPU. Retain a measured memory cap and chunk fallback. |
| B4 | Extend one validated scalar device path through the required response and likelihood. | High effort; value depends on B0. | Warm, synchronized timing and transfer counts beat the matched CPU path; boundary proposals and numerical tolerances pass. Run a bounded repeated-call memory test before a retrieval. |
| B5 | Test mixed precision and multiple GPU workers after single-device parity. | High effort; conditional value. | Measure weak opacity, saturated lines, cancellation, spectra, log likelihood, evidence, and posterior/noise-ensemble effects. Bind processes within allocated devices and measure full sampler wall time. |
| B6 | Compile or compose fixed broadening operators if B0 identifies a response cost. | Conditional value; medium effort. | Match the sequential Gray/Gaussian response, edge behavior, and normalization. Measure warm time and memory before choosing an implementation. |

Keep MultiNest as the production nested sampler. The new paper does not by
itself justify adding another sampler. If proposal supply is the measured
limit, assess a batch interface in a separate research task with evidence and
posterior gates. More device throughput alone is not a retrieval result.

For multiple GPUs, verify device identity for every worker. The inspected
BeAR core does not establish how the launcher assigns a unique GPU to each
rank; ROBERT's CUDA runtime also chooses from visible devices. Record explicit
device visibility/assignment and resource use. Do not assume that starting
more MPI ranks gives more independent GPUs.

For fixed-vector acceleration comparisons, use identical input data, priors,
opacity, wavelength samples, atmospheric layers, RT, masks, response, and
likelihood definition. Set a case-specific numerical tolerance before coding.
Report median and tail timing, compilation cost, memory, and failed/invalid
proposal classification. Repeat posterior checks only for candidates that pass
this first gate.

For a changed method such as a fixed filter or reduced precision, also require
injection recovery, several seeds/noise draws, and stable abundance-ratio and
evidence results. Keep existing checkpoints and run revisions unchanged. Use
the required cluster queue for compute-heavy tests.

## Decision

The immediate candidates are a fixed-filter HRS mode and a smaller complete
CPU likelihood. A fused sampled-opacity GPU path is a useful next research
experiment if profiling supports it. The present Virga cloud study must first
establish converged recovery; the HRS speed result cannot establish a speed-up
for cloud formation or correlated-k SH4 retrievals.

No BeAR implementation was copied into ROBERT. The source retains its own GPL
license; any ROBERT implementation in this plan must be independently written.
