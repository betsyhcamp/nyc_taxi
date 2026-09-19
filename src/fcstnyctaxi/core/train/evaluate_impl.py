import logging

import pandas as pd

from fcstnyctaxi.lib.column_checks import require_columns
from fcstnyctaxi.lib.period_utils import ORIGIN_TIME_UNIT, derive_horizon_label

_log = logging.getLogger(__name__)

_FOLD_KEYS = ["forecast_origin_date", "predicted_fiscal_year_month"]
_JOIN_KEYS = _FOLD_KEYS + ["unique_id"]

# The columns evaluate takes off each sidecar's monthly_series.parquet.
_MONTHLY_SERIES_COLUMNS = _JOIN_KEYS + [
    "tier",
    "series_weight",
    "monthly_forecast",
    "actual_monthly_total",
    "origin_month_fraction_elapsed",
]
# The benchmark contributes its forecast plus the three columns both sides share.
_BENCHMARK_COLUMNS = _JOIN_KEYS + [
    "monthly_forecast",
    "tier",
    "series_weight",
    "actual_monthly_total",
]

_BASE_FRAME_COLUMNS = _JOIN_KEYS + [
    "origin_fiscal_year_month",
    "origin_month_fraction_elapsed",
    "horizon",
    "tier",
    "series_weight",
    "monthly_forecast_ch",
    "monthly_forecast_bm",
    "actual_monthly_total",
    "_merge",
    "challenger_model",
    "benchmark_model",
]


def present_tier_labels(
    tier_values: pd.Series, tier_labels: tuple[str, ...]
) -> tuple[str, ...]:
    """The configured labels this run's data actually uses, in configured order.

    `assign_tiers` bins `tier_labels[:effective_tiers]`, so config would name
    tiers no series has. Read off values, not `.cat.categories`, since a valid
    sidecar can carry a string `tier`. Nulls are kept: one would vanish from every
    tier slice while still counting in global.

    Raises:
        ValueError: If the values are not a prefix of `tier_labels`, meaning the
            sidecar was tiered under config this run does not declare.
    """
    observed = set(pd.Series(tier_values).astype(str).unique())
    present = tuple(label for label in tier_labels if label in observed)

    if set(present) != observed or present != tuple(tier_labels[: len(present)]):
        raise ValueError(
            f"tier values {sorted(observed)} are not a prefix of the configured "
            f"tier_labels {list(tier_labels)}: assign_tiers only ever emits a "
            "prefix, so the sidecar was built under different tiering config."
        )
    return present


def _as_ordered_tier(tier_values: pd.Series, present: tuple[str, ...]) -> pd.Series:
    """Rebuild `tier` as an ordered categorical over the present labels.

    A concatenated categorical degrades to `object` when the per-origin category
    sets differ, which `assign_tiers` produces legally, and such a column sorts
    alphabetically. One shared list is also what lets the two sides be compared.
    """
    values = pd.Series(tier_values)
    return pd.Series(
        pd.Categorical(values.astype(str), categories=present, ordered=True),
        index=values.index,
        name="tier",
    )


def _origin_fiscal_month_lookup(calendar_df: pd.DataFrame) -> pd.Series:
    """The origin-to-fiscal-month map, deduped, indexed by origin date.

    Checked here, not in `_check_base_frame`: one `ds` carrying two fiscal months
    fans the join out before anything can look at it. Re-expressed from
    `label_horizon`, which discards the map its caller needs as a column.
    """
    require_columns(calendar_df, ["ds", "fiscal_year_month"], "calendar_df")
    lookup = calendar_df[["ds", "fiscal_year_month"]].drop_duplicates()

    if not lookup["ds"].is_unique:
        conflicting = lookup.loc[lookup["ds"].duplicated(), "ds"].unique()
        raise ValueError(
            f"calendar maps {len(conflicting)} ds value(s) to more than one "
            f"fiscal_year_month; first few: {sorted(conflicting)[:5]}"
        )

    lookup = lookup.assign(ds=lookup["ds"].astype(f"datetime64[{ORIGIN_TIME_UNIT}]"))
    return lookup.set_index("ds")["fiscal_year_month"]


def _disagreeing(left: pd.Series, right: pd.Series) -> pd.Series:
    """Rows where the two sides differ. Both-null counts as agreement: `!=` reads
    a pair of nulls as a mismatch and would send an operator upstream."""
    return (left != right) & ~(left.isna() & right.isna())


def _check_base_frame(base: pd.DataFrame) -> None:
    """The checks on the joined frame, before its `_ch`/`_bm` collapse.

    The calendar's uniqueness check fires earlier, in `_origin_fiscal_month_lookup`.

    Raises:
        ValueError: On a one-sided join row, a side disagreeing on tier, weight or
            actual, an actual that varies by origin, or an origin absent from the
            calendar.
    """
    # Both sides matched. The relative metrics merge the two sides internally and
    # the absolute ones never merge, so the families run over different row sets
    # with nothing enforcing they agree. Also the only check on a key that looks
    # equal and is not: pandas bridges datetime units and raises on a dtype
    # mismatch, so only same-dtype keys that differ (a stray time component) do.
    one_sided = base["_merge"] != "both"
    if one_sided.any():
        counts = base.loc[one_sided, "_merge"].value_counts()
        sample = base.loc[one_sided, _JOIN_KEYS].head(5)
        raise ValueError(
            f"{int(one_sided.sum())} of {len(base)} comparison row(s) matched one "
            f"side only {counts[counts > 0].to_dict()}: left_only is a challenger "
            f"row the benchmark lacks, right_only the reverse.\n"
            f"{sample.to_string(index=False)}"
        )

    # The two sidecars agree on the three columns both compute from one panel under
    # one config, so a disagreement is an upstream fault. Categories first: two
    # categoricals that do not share them raise a TypeError.
    if list(base["tier_ch"].cat.categories) != list(base["tier_bm"].cat.categories):
        raise ValueError(
            "the two sidecars' tier categories differ: "
            f"{list(base['tier_ch'].cat.categories)} against "
            f"{list(base['tier_bm'].cat.categories)}."
        )
    for column in ("tier", "series_weight", "actual_monthly_total"):
        mismatched = _disagreeing(base[f"{column}_ch"], base[f"{column}_bm"])
        if mismatched.any():
            sample = base.loc[mismatched, _JOIN_KEYS].head(5)
            raise ValueError(
                f"{int(mismatched.sum())} row(s) disagree on {column!r} across the "
                f"two sidecars, which both derive it from the same panel:\n"
                f"{sample.to_string(index=False)}"
            )

    # A cross-step check on backtest: a realized total cannot depend on its origin.
    per_series_month = base.groupby(
        ["unique_id", "predicted_fiscal_year_month"], observed=True
    )["actual_monthly_total_ch"].nunique(dropna=False)
    varying = per_series_month[per_series_month > 1]
    if not varying.empty:
        raise ValueError(
            f"{len(varying)} series-month(s) carry more than one "
            f"actual_monthly_total across origins; first few:\n"
            f"{varying.head(5).to_string()}"
        )

    # Every origin is in the calendar. derive_horizon_label emits "horizon_nan" on
    # a null rather than raising, so a wrong label would reach every table.
    absent = base["origin_fiscal_year_month"].isna()
    if absent.any():
        missing = base.loc[absent, "forecast_origin_date"].unique()
        raise ValueError(
            f"{len(missing)} forecast origin(s) absent from the calendar; first "
            f"few: {sorted(missing)[:5]}"
        )


def _build_base_frame(
    *,
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    challenger_model: str,
    benchmark_model: str,
    tier_labels: tuple[str, ...],
) -> pd.DataFrame:
    """Challenger joined to benchmark, one row per origin, month and series.

    Outer with an indicator rather than inner, so the metric functions' row-set
    assumption becomes a filterable column plus a check. Wide rather than long by
    model, because a ratio needs both forecasts on one row.

    `forecast_origin_date` is normalized on both sides and on the calendar for the
    output contract, not the join: pandas bridges the units, but a merged key
    inherits the left side's.
    """
    require_columns(challenger_ms, _MONTHLY_SERIES_COLUMNS, "challenger monthly_series")
    require_columns(benchmark_ms, _BENCHMARK_COLUMNS, "benchmark monthly_series")

    present = present_tier_labels(
        pd.concat([challenger_ms["tier"], benchmark_ms["tier"]], ignore_index=True),
        tier_labels,
    )

    challenger = challenger_ms[_MONTHLY_SERIES_COLUMNS].assign(
        forecast_origin_date=challenger_ms["forecast_origin_date"].astype(
            f"datetime64[{ORIGIN_TIME_UNIT}]"
        ),
        tier=_as_ordered_tier(challenger_ms["tier"], present),
    )
    benchmark = benchmark_ms[_BENCHMARK_COLUMNS].assign(
        forecast_origin_date=benchmark_ms["forecast_origin_date"].astype(
            f"datetime64[{ORIGIN_TIME_UNIT}]"
        ),
        tier=_as_ordered_tier(benchmark_ms["tier"], present),
    )

    base = challenger.merge(
        benchmark, on=_JOIN_KEYS, suffixes=("_ch", "_bm"), how="outer", indicator=True
    )
    base["origin_fiscal_year_month"] = base["forecast_origin_date"].map(
        _origin_fiscal_month_lookup(calendar_df)
    )

    _check_base_frame(base)

    # The two sides agree, so keep one of each pair. The challenger's, as subject.
    base = base.rename(
        columns={
            "tier_ch": "tier",
            "series_weight_ch": "series_weight",
            "actual_monthly_total_ch": "actual_monthly_total",
        }
    )
    base["horizon"] = derive_horizon_label(
        predicted_fiscal_year_month=base["predicted_fiscal_year_month"],
        origin_fiscal_year_month=base["origin_fiscal_year_month"],
        origin_month_fraction_elapsed=base["origin_month_fraction_elapsed"],
    )
    base["challenger_model"] = challenger_model
    base["benchmark_model"] = benchmark_model

    # No derived error columns: inputs only, so the near-zero guard on any ratio
    # stays a consumer's decision rather than the substrate's.
    return base[_BASE_FRAME_COLUMNS].sort_values(_JOIN_KEYS).reset_index(drop=True)
