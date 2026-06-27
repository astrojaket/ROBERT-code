# Developer Guidelines

These guidelines define how contributors should work in ROBERT.

## 1. Contributor Contract

Every substantial contribution must:

- Follow RFC-0001.
- Preserve package dependency rules.
- Add tests for public behavior.
- Document scientific assumptions.
- Avoid hidden global state.
- Keep optional dependencies optional.

## 2. Design Before Implementation

Before adding a major feature, answer:

1. Which package owns this feature?
2. Which public interface does it implement?
3. What data model objects does it consume and produce?
4. What tests validate it?
5. What scientific assumptions and citations apply?
6. Does it require an architecture doc update?

If the feature changes public interfaces, configuration schema, result schema,
or dependency rules, update RFC companion docs in the same PR.

## 3. Coding Style

Rules:

- Prefer small modules and small functions.
- Use typed public APIs.
- Use NumPy-style docstrings.
- Use clear names over clever names.
- Keep comments for non-obvious scientific or architectural choices.
- Do not add physics-looking placeholders unless clearly labeled.

Avoid:

- Broad utility modules with unrelated functions.
- Long positional argument lists.
- Raw config dictionaries outside `io.config`.
- Hidden mutation of arrays stored in domain objects.
- Import-time side effects.

## 4. Public API Rules

Public APIs must:

- Be typed.
- Have docstrings.
- Raise clear errors.
- State units and array shape conventions.
- Avoid exposing backend-specific internals.

Private helpers:

- Use a leading underscore.
- May change without deprecation before v1.0.
- Must not be imported by examples or tutorials.

## 5. Scientific Components

Every scientific component must declare:

- Purpose.
- Required parameters.
- Units.
- Validity range.
- Assumptions.
- Citations.
- Test/validation strategy.

Examples:

- A temperature profile declares required parameters and pressure-grid
  assumptions.
- A cloud model declares optical-property assumptions and valid particle-size
  domain.
- An opacity provider declares coverage and interpolation behavior.

## 6. Error Handling

Use clear exceptions:

- Configuration problems: config error.
- Data problems: data error.
- Opacity/coverage problems: coverage error.
- Invalid runtime state: runtime error.
- Programmer mistakes: standard Python exceptions.

Error messages should include:

- Component name.
- Parameter or path involved.
- What was expected.
- What was received.

## 7. Logging

Rules:

- Use `logging.getLogger(__name__)`.
- Do not configure global logging at import.
- CLI configures log level and handlers.
- Warnings that affect science should be captured in the manifest.

Use logging for:

- Setup decisions.
- Cache hits/misses at debug level.
- Long-running progress.
- Validation warnings.

Do not use logging for:

- Returning data.
- Hiding errors.
- Replacing tests.

## 8. Optional Dependencies

Optional dependencies belong in extras and adapter modules.

Rules:

- Import optional packages inside the adapter that needs them.
- Provide a helpful error message if missing.
- Do not make core imports fail because an optional sampler/GPU package is not
  installed.

## 9. Configuration Changes

Config changes require:

- Schema update.
- Documentation update.
- Test coverage.
- Migration note if existing configs break.

Unknown fields should fail validation unless they live under an explicit plugin
extension namespace.

## 10. Result Schema Changes

Result schema changes require:

- Schema version bump.
- Reader/writer tests.
- Migration or compatibility note.

Retrieval results must always preserve:

- Parameter names.
- Samples/posterior representation.
- Best-fit or representative prediction.
- Manifest.
- ROBERT version.

## 11. Review Checklist

Reviewers should ask:

- Does the change belong in this package?
- Are dependencies allowed?
- Are public interfaces documented and typed?
- Are tests appropriate for risk?
- Is scientific validation adequate?
- Are optional dependencies isolated?
- Does the run manifest capture new science-relevant settings?
- Does the contribution avoid import-time side effects?

## 12. Documentation Expectations

For new features, update:

- API docstrings.
- Theory docs for equations/assumptions.
- User docs if the feature is user-facing.
- Developer docs if the feature changes architecture or extension points.

Examples and tutorials must use public APIs only.
