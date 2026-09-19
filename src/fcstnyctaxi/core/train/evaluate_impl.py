import logging
from collections.abc import Callable, Iterator
from itertools import product
from typing import Any

import pandas as pd

from fcstnyctaxi.lib.column_checks import require_columns
from fcstnyctaxi.lib.fold_metrics import (
    compute_signed_bias_per_series,
    compute_signed_bias_pooled,
    compute_wape,
    compute_weighted_signed_bias,
    compute_wrmae_per_series,
    compute_wrmae_pooled,
)
from fcstnyctaxi.lib.period_utils import ORIGIN_TIME_UNIT, derive_horizon_label

_log = logging.getLogger(__name__)

_FOLD_KEYS = ["forecast_origin_date", "predicted_fiscal_year_month"]
_JOIN_KEYS = _FOLD_KEYS + ["unique_id"]
_SCORE_KEYS = ["model", "horizon", "tier", "metric"]

GLOBAL_TIER = "global"
"""The aggregate row beside the tier partition, so never sum across `tier`."""

# The vocabulary is the function name minus `compute_`, so a seventh names itself.
# Two mappings rather than one adapter: the two call shapes below are real.
_RELATIVE_METRIC_FNS: dict[
    str, Callable[[pd.DataFrame, pd.DataFrame, str | None], float]
] = {
    "wrmae_pooled": compute_wrmae_pooled,
    "wrmae_per_series": compute_wrmae_per_series,
}
_ABSOLUTE_METRIC_FNS: dict[str, Callable[[pd.DataFrame, str | None], float]] = {
    "wape": compute_wape,
    "weighted_signed_bias": compute_weighted_signed_bias,
    "signed_bias_pooled": compute_signed_bias_pooled,
    "signed_bias_per_series": compute_signed_bias_per_series,
}
_METRIC_NAMES = tuple(_RELATIVE_METRIC_FNS) + tuple(_ABSOLUTE_METRIC_FNS)

# What the metric functions consume, once a side is narrowed to one forecast.
# `horizon` rides along so a caller can score a summary-grain slice of a view.
_VIEW_COLUMNS = _JOIN_KEYS + [
    "monthly_forecast",
    "actual_monthly_total",
    "series_weight",
    "tier",
    "horizon",
]

_FOLD_GRAIN = ["model", "horizon", "tier"] + _FOLD_KEYS
_FOLD_METRIC_COLUMNS = _FOLD_GRAIN + ["metric", "value", "n_obs"]

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


def _model_view(base: pd.DataFrame, forecast_column: str) -> pd.DataFrame:
    """One model's slice of the wide base frame, in the shape the metrics take."""
    return base.rename(columns={forecast_column: "monthly_forecast"})[_VIEW_COLUMNS]


def _global_first_tier(tier_values: pd.Series, present: pd.Index) -> pd.Series:
    """`tier` on a score table: ordered, with the `global` aggregate row first.

    A dashboard reading only summary_metrics has no other ordering source, and a
    string column sorts high, low, middle, very_high, very_low.
    """
    return pd.Series(
        pd.Categorical(tier_values, categories=[GLOBAL_TIER, *present], ordered=True),
        index=tier_values.index,
        name="tier",
    )


def _model_folds(
    model_views: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> Iterator[tuple[str, tuple, pd.DataFrame, pd.DataFrame]]:
    """Every (model, fold) pair, both sides already cut to that fold.

    Split out because the axes are not independent: a fold slice needs its model,
    so the groupby is per-model setup. Inlining it wedges two statements between
    two `for` statements.
    """
    for model, (challenger, benchmark) in model_views.items():
        challenger_folds = dict(tuple(challenger.groupby(_FOLD_KEYS, observed=True)))
        benchmark_folds = dict(tuple(benchmark.groupby(_FOLD_KEYS, observed=True)))
        for fold_key, challenger_fold in challenger_folds.items():
            yield model, fold_key, challenger_fold, benchmark_folds[fold_key]


def _score_folds(
    base: pd.DataFrame, *, challenger_model: str, benchmark_model: str
) -> pd.DataFrame:
    """Every metric for both models at fold grain, in long form.

    Both models run through the same six with no branch, so the benchmark scores
    itself and reads 1.0: no conditional, a square metric set under `pivot_table`,
    and the reference line on a skill chart. Never test it with `== 1.0`, since
    `wrmae_per_series` renormalizes weights and lands 1.0 to float tolerance.

    `n_obs` counts rows available before metric-specific exclusions, not rows used.
    Rows used would need all six functions to return a count.
    """
    challenger_view = _model_view(base, "monthly_forecast_ch")
    benchmark_view = _model_view(base, "monthly_forecast_bm")

    # A model in both roles is a legitimate smoke test and must score once: two
    # identical row sets under one name would double every n_obs.
    model_views = {challenger_model: (challenger_view, benchmark_view)}
    model_views.setdefault(benchmark_model, (benchmark_view, benchmark_view))

    tier_slices: list[tuple[str, str | None]] = [(GLOBAL_TIER, None)] + [
        (label, label) for label in base["tier"].cat.categories
    ]
    # Horizon is a function of the origin alone, so every fold has exactly one,
    # which is also why horizon does not multiply this table's row count.
    fold_horizons = base.drop_duplicates(_FOLD_KEYS).set_index(_FOLD_KEYS)["horizon"]

    rows = []
    for fold, tier_slice in product(_model_folds(model_views), tier_slices):
        model, fold_key, challenger_fold, benchmark_fold = fold
        tier_label, tier_value = tier_slice
        origin, predicted_month = fold_key

        row: dict[str, Any] = {
            "model": model,
            "horizon": fold_horizons.loc[fold_key],
            "tier": tier_label,
            "forecast_origin_date": origin,
            "predicted_fiscal_year_month": predicted_month,
            "n_obs": len(challenger_fold)
            if tier_value is None
            else int((challenger_fold["tier"] == tier_value).sum()),
        }
        for name, relative_fn in _RELATIVE_METRIC_FNS.items():
            row[name] = relative_fn(challenger_fold, benchmark_fold, tier_value)
        for name, absolute_fn in _ABSOLUTE_METRIC_FNS.items():
            row[name] = absolute_fn(challenger_fold, tier_value)
        rows.append(row)

    long = pd.DataFrame(rows).melt(
        id_vars=_FOLD_GRAIN + ["n_obs"],
        value_vars=list(_METRIC_NAMES),
        var_name="metric",
        value_name="value",
    )
    long["tier"] = _global_first_tier(long["tier"], base["tier"].cat.categories)
    return (
        long[_FOLD_METRIC_COLUMNS]
        .sort_values(_SCORE_KEYS + _FOLD_KEYS)
        .reset_index(drop=True)
    )
