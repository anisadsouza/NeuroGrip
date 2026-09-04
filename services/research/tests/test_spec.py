import pytest

from neurogrip.spec import (
    AR_ORDER,
    PER_CHANNEL_FEATURES,
    PSD_BINS,
    feature_count,
    feature_names,
)


def test_per_channel_table_is_the_documented_seventeen():
    assert PER_CHANNEL_FEATURES == (
        "rms", "mav", "wl", "zc", "ssc",
        "ar1", "ar2", "ar3", "ar4",
        "psd1", "psd2", "psd3", "psd4", "psd5", "psd6",
        "mdf", "mnf",
    )
    assert len(PER_CHANNEL_FEATURES) == 5 + AR_ORDER + PSD_BINS + 2


def test_feature_names_are_channel_major():
    names = feature_names(2)
    assert names[0] == "ch1_rms"
    assert names[16] == "ch1_mnf"
    assert names[17] == "ch2_rms"
    assert len(names) == 34


def test_ordering_is_numeric_not_lexicographic_at_ten_channels():
    """sorted() places ch10_rms before ch2_rms. This is the bug being fixed."""
    names = feature_names(12)
    assert names.index("ch2_rms") < names.index("ch10_rms")
    assert names != tuple(sorted(names))


def test_feature_count_matches_names():
    for n in (1, 2, 8, 12, 16):
        assert feature_count(n) == len(feature_names(n))


def test_rejects_non_positive_channel_counts():
    with pytest.raises(ValueError):
        feature_names(0)
    with pytest.raises(ValueError):
        feature_count(0)
