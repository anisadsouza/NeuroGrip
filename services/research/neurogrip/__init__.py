"""NeuroGrip research toolkit: sEMG simulation, features, training, evaluation."""

__all__ = ["__version__", "FEATURE_SPEC_VERSION"]

__version__ = "0.2.0"

# Bump whenever the feature set, its ordering, or any DSP constant changes.
# packages/core/src/spec.ts must be bumped in lockstep and fixtures regenerated.
FEATURE_SPEC_VERSION = 1
