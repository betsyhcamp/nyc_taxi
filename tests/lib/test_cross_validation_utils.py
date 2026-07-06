from __future__ import annotations

import pandas as pd
import pytest

from fcstnyctaxi.lib.cross_validation_utils import sorted_origin_horizon_pairs

# ================================================
# sorted_origin_horizon_pairs — happy path
# ================================================


def test_sorted_origin_horizon_pairs_already_sorted_normalizes_to_timestamp() -> None:
    """Already-sorted string origins are returned in order, cast to pd.Timestamp."""
    pairs = [
        ("2025-04-20", 2),
        ("2025-04-27", 1),
    ]
    result = sorted_origin_horizon_pairs(pairs, freq="W")

    assert result == [
        (pd.Timestamp("2025-04-20"), 2),
        (pd.Timestamp("2025-04-27"), 1),
    ]


def test_sorted_origin_horizon_pairs_freq_1_casts_origins_to_int() -> None:
    """When freq == 1, string origins are cast to int."""
    pairs = [("1", 2), ("2", 1)]
    result = sorted_origin_horizon_pairs(pairs, freq=1)

    assert result == [(1, 2), (2, 1)]


# ================================================
# sorted_origin_horizon_pairs — sorting behavior
# ================================================


@pytest.mark.parametrize(
    "freq, pairs, expected",
    [
        (
            "W",
            [("2025-04-27", 1), ("2025-04-20", 2)],
            [(pd.Timestamp("2025-04-20"), 2), (pd.Timestamp("2025-04-27"), 1)],
        ),
        (
            1,
            [(2, 1), (1, 2)],
            [(1, 2), (2, 1)],
        ),
    ],
    ids=["freq_timestamp", "freq_integer"],
)
def test_sorted_origin_horizon_pairs_out_of_order_input_is_sorted(
    freq, pairs, expected
) -> None:
    """Out-of-order origins are returned in ascending chronological order."""
    result = sorted_origin_horizon_pairs(pairs, freq)
    assert result == expected


# ================================================
# sorted_origin_horizon_pairs — edge cases
# ================================================


def test_sorted_origin_horizon_pairs_single_pair_returns_list_with_one_tuple() -> None:
    """A single-element list is returned as a single-element list."""
    pairs = [("2025-04-20", 2)]
    result = sorted_origin_horizon_pairs(pairs, freq="W")

    assert result == [(pd.Timestamp("2025-04-20"), 2)]
