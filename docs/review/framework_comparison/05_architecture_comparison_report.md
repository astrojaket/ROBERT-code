# Architecture Comparison Report

## NEMESIS

Exceptionally well: NEMESIS preserves decades of validated retrieval practice,
especially optimal-estimation diagnostics, contribution functions, scattering,
clouds, and geometry handling.

Elegant decisions: a recognizable separation between retrieval iteration,
forward model, opacity data, and diagnostics exists even inside an older
Fortran architecture.

ROBERT should adopt: diagnostic products, reference examples, explicit
covariance/averaging-kernel concepts, and a validation culture tied to known
atmospheric cases.

Unnecessarily complicated: run-name sidecar files, common-block state, compile
time dimensions, integer mode flags, and separate executable variants.

Do not repeat: files as internal function arguments, hidden global state, and
duplicated variants for each geometry.

## NemesisPy

Exceptionally well: NemesisPy shows which NEMESIS concepts are useful for modern
exoplanet emission, transmission, and phase-curve workflows.

Elegant decisions: Numba-friendly kernels and Python examples make prototyping
faster than the original Fortran workflow.

ROBERT should adopt: compact exoplanet examples, Python-first data objects, and
Numba as an optional backend for hot loops.

Unnecessarily complicated: raw dictionaries, mutable backend objects that require
specific call order, and sampler code outside a mature retrieval package.

Do not repeat: letting examples become the real retrieval API.

## TauREx

Exceptionally well: TauREx has one of the strongest modern Python architectures:
class discovery, component factories, model contributions, fit-parameter
introspection, tests, docs, CI, and optional sampler extras.

Elegant decisions: optical-depth contributions can be added to a forward model
without rewriting the whole model. Components expose fitting and derived
parameters.

ROBERT should adopt: a small contribution interface, output/citation hooks,
component registries, and sampler adapters that consume a stable problem object.

Unnecessarily complicated: factory/mixin composition is powerful but can become
opaque for users and maintainers.

Do not repeat: plugin cleverness before ROBERT has a small, stable internal
protocol.

## POSEIDON

Exceptionally well: POSEIDON covers modern exoplanet retrieval needs: stellar
contamination, high-resolution likelihoods, multidimensional atmospheres,
transmission, emission, reflection, chemistry grids, and detailed instruments.

Elegant decisions: expensive stellar spectra, chemistry grids, and instrument
structures are precomputed before sampler calls.

ROBERT should adopt: explicit instrument models, stellar contamination as a
first-class optional model, and early support for multi-instrument calibration
parameters.

Unnecessarily complicated: broad dict-based run functions with many arguments,
import-time MPI/environment choices, and tight coupling of retrieval orchestration
to one sampler family.

Do not repeat: global MPI communicator assumptions and thread settings at import.

## petitRADTRANS

Exceptionally well: petitRADTRANS draws a useful boundary around a reusable RT
object. It supports c-k and line-by-line modes, clouds, CIA, Rayleigh, scattering,
and retrieval datasets with covariance/photometry/offset concerns.

Elegant decisions: `Radtrans` owns loaded opacity state and warns against unsafe
mutation; `Data` represents one observation/instrument with its own resolution
and transformation details.

ROBERT should adopt: immutable or rebuild-only RT engines, dataset objects, HDF5
or similarly structured opacity stores, and strong reference tests.

Unnecessarily complicated: broad spectral-model configuration can accumulate
model topology, observation, retrieval, and plotting concerns in one layer.

Do not repeat: making the highest-level convenience object the core domain model.

## PICASO

Exceptionally well: PICASO is unusually strong in reflected light, scattering,
phase geometry, climate workflows, Virga clouds, and reference-data curation.

Elegant decisions: opacity factory workflows and reference-data version checks
make external data part of the scientific contract.

ROBERT should adopt: reference-data manifests, explicit phase/geometry objects,
and climate/chemistry coupling only as optional backends.

Unnecessarily complicated: import-time environment checks can prevent partial use,
and top-level functions carry many responsibilities.

Do not repeat: hard failure on optional data at import; validate when a feature is
constructed instead.

## CHIMERA

Exceptionally well: CHIMERA makes research workflows approachable through
concrete HST/JWST notebooks and templates, and it documents free vs chemically
consistent retrieval patterns.

Elegant decisions: practical examples expose how a scientist actually modifies
temperature profiles, chemistry, clouds, and samplers.

ROBERT should adopt: executable tutorial cases and explicit example families for
clear, cloudy, free-chemistry, and equilibrium-chemistry retrievals.

Unnecessarily complicated: notebooks/scripts are the primary control surface,
with generated outputs and plotting artifacts mixed into the repository.

Do not repeat: treating user-modified scripts as configuration.

## Brewster

Exceptionally well: Brewster is focused on brown-dwarf and giant-planet emission
retrievals, with expressive cloud/patch models, calibration factors, and Fortran
RT kernels.

Elegant decisions: patchy clouds and calibration nuisance parameters are made
explicit in retrieval templates, and benchmark spectra support installation
checks.

ROBERT should adopt: clear nuisance-parameter handling for multi-segment spectra,
patch/column abstractions, and a simple forward-model benchmark command.

Unnecessarily complicated: hard-coded local paths, old pinned dependencies,
global settings tuples, and template files that users must edit directly.

Do not repeat: global `settings.runargs` or dependency pinning to obsolete runtime
versions as the only supported path.

## Exo_Skryer

Exceptionally well: Exo_Skryer is a modern performance-oriented design: JAX
kernels, YAML schema, registries for physics choices, many sampler adapters,
offset groups, JAX-jitted likelihoods, OS/CK opacity, and 1.5D transit support.

Elegant decisions: the kernel registry maps YAML choices to pure functions. Each
sampler gets a self-contained adapter around the same forward-model callable.

ROBERT should adopt: explicit configuration schema, registry-backed kernel
selection, offset-group conventions, and sampler adapters with consistent output
to ArviZ-like products.

Unnecessarily complicated: many samplers and backends appear before documentation
and tests are equally mature; global opacity registries can hide run state.

Do not repeat: growing backend breadth faster than validation depth.

## Cross-Cutting Architecture Ranking

Most adoptable for ROBERT now:

1. petitRADTRANS dataset and RT-engine boundary.
2. TauREx contribution/registry/fitting-parameter pattern.
3. Exo_Skryer YAML schema and sampler-adapter shape.
4. POSEIDON instrument and stellar-contamination precomputation.
5. NEMESIS diagnostic and validation culture.

Most dangerous to copy directly:

1. NEMESIS sidecar-file internals.
2. POSEIDON import-time MPI/thread mutation.
3. CHIMERA/Brewster template-as-API workflows.
4. PICASO import-time reference-data hard failures.
5. Any framework's unchecked global opacity/cache state.
