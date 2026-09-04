"""Canonical feature specification.

The ONLY place the feature set and its ordering are declared.
`packages/core/src/spec.ts` mirrors this exactly and
`packages/core/test/conformance.test.ts` asserts they agree. Any change here
requires bumping `neurogrip.FEATURE_SPEC_VERSION` and regenerating fixtures.
"""

from __future__ import annotations

AR_ORDER = 4
PSD_BINS = 6

PER_CHANNEL_FEATURES: tuple[str, ...] = (
    # Time domain
    "rms",   # root mean square
    "mav",   # mean absolute value
    "wl",    # waveform length
    "zc",    # zero crossings
    "ssc",   # slope sign changes
    # Autoregressive coefficients (Levinson-Durbin over biased autocorrelation)
    *(f"ar{i + 1}" for i in range(AR_ORDER)),
    # Power spectral density, linearly spaced bands over the one-sided spectrum
    *(f"psd{i + 1}" for i in range(PSD_BINS)),
    # Spectral shape. Consumed downstream by the fatigue and drift estimators:
    # median frequency falls as a muscle fatigues.
    "mdf",
    "mnf",
)


def feature_names(n_channels: int) -> tuple[str, ...]:
    """Full ordering: channel-major, fixed intra-channel order.

    Deliberately NOT lexicographic. sorted() emits ch10_* before ch2_*, which
    silently permutes the vector for any recording with ten or more channels.
    """
    if n_channels < 1:
        raise ValueError("n_channels must be at least 1")
    return tuple(
        f"ch{channel}_{feature}"
        for channel in range(1, n_channels + 1)
        for feature in PER_CHANNEL_FEATURES
    )


def feature_count(n_channels: int) -> int:
    if n_channels < 1:
        raise ValueError("n_channels must be at least 1")
    return n_channels * len(PER_CHANNEL_FEATURES)
