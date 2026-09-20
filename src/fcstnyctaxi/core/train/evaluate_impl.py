import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

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
from fcstnyctaxi.schemas.config.train import ModelRoles, TrainModelingConfig
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity

_log = logging.getLogger(__name__)

_FOLD_KEYS = ["forecast_origin_date", "predicted_fiscal_year_month"]
_JOIN_KEYS = _FOLD_KEYS + ["unique_id"]
_SCORE_KEYS = ["model", "horizon", "tier", "metric"]
_PERIOD_KEYS = _SCORE_KEYS + ["predicted_fiscal_year_month"]

GLOBAL_TIER = "global"
"""The aggregate row beside the tier partition, so never sum across `tier`."""
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

# Read off the calendar per origin, not off monthly_series: backtest transcribes
# the second of these into every row, and horizon labeling turns on its value.
_ORIGIN_ATTRIBUTES = ["fiscal_year_month", "origin_month_fraction_elapsed"]

_FOLD_GRAIN = ["model", "horizon", "tier"] + _FOLD_KEYS
_FOLD_METRIC_COLUMNS = _FOLD_GRAIN + ["metric", "value", "n_obs"]

# The columns evaluate takes off each sidecar's monthly_series.parquet.
_MONTHLY_SERIES_COLUMNS = _JOIN_KEYS + [
    "tier",
    "series_weight",
    "monthly_forecast",
    "actual_monthly_total",
]
# The benchmark contributes its forecast plus the three columns both sides share.
_BENCHMARK_COLUMNS = _JOIN_KEYS + [
    "monthly_forecast",
    "tier",
    "series_weight",
    "actual_monthly_total",
]

# The suffixed columns every metric reads. The two forecasts are never compared
# to each other, and a matched pair of nulls passes the agreement check below.
_METRIC_INPUT_COLUMNS = [
    "monthly_forecast_ch",
    "monthly_forecast_bm",
    "actual_monthly_total_ch",
    "series_weight_ch",
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

# EvaluateOutputs field to output file; the write loop and output_rows share it.
_OUTPUT_FILENAMES = {
    "per_series_comparison": "per_series_comparison.parquet",
    "fold_metrics": "fold_metrics.parquet",
    "period_metrics": "period_metrics.parquet",
    "summary_metrics": "summary_metrics.parquet",
}
_MANIFEST_FILENAME = "evaluate_manifest.json"
_STEP_DIR_NAME = "evaluate"

# The three sidecar files evaluate opens. Not metrics.parquet and nothing at
# weekly grain: a dashboard wanting weekly plots reads the sidecar itself.
_SIDECAR_MANIFEST = "backtest_manifest.json"
_SIDECAR_MONTHLY_SERIES = "monthly_series.parquet"
_SIDECAR_CALENDAR = "fiscal_calendar.parquet"

_HERO_METRIC_NAME = "wrmae_pooled"


@dataclass(frozen=True)
class EvaluateOutputs:
    """The four frames the step persists; the field names are the filenames."""

    per_series_comparison: pd.DataFrame
    fold_metrics: pd.DataFrame
    period_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame


@dataclass(frozen=True)
class EvaluateSummary:
    """What this run scored, for a wrapper that never opens the tables. The hero
    values keep NaN; `as_dict` and `_build_manifest` are where it becomes `None`."""

    train_run_id: str
    challenger_model: str
    benchmark_model: str
    feature_run_id: str
    n_origins: int
    first_origin: str
    last_origin: str
    n_series: int
    n_folds_total: int
    hero_metric_name: str
    hero_metric_values: dict[str, float | None]
    output_rows: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        """Coerce to JSON-safe primitives: a numpy scalar or a NaN breaks KFP."""
        return {
            "train_run_id": str(self.train_run_id),
            "challenger_model": str(self.challenger_model),
            "benchmark_model": str(self.benchmark_model),
            "feature_run_id": str(self.feature_run_id),
            "n_origins": int(self.n_origins),
            "first_origin": str(self.first_origin),
            "last_origin": str(self.last_origin),
            "n_series": int(self.n_series),
            "n_folds_total": int(self.n_folds_total),
            "hero_metric_name": str(self.hero_metric_name),
            "hero_metric_values": _json_floats(self.hero_metric_values),
            "output_rows": {name: int(rows) for name, rows in self.output_rows.items()},
        }


def _json_floats(values: dict[str, float | None]) -> dict[str, float | None]:
    """NaN to None on both paths out, since the dataclass flattens where the
    manifest nests. `json.dumps` writes a bare `NaN` only a strict reader rejects."""
    return {
        key: None if value is None or pd.isna(value) else float(value)
        for key, value in values.items()
    }


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
    """Rebuild `tier` as an ordered categorical over the present labels: a
    concatenated one degrades to `object` when the per-origin category sets
    differ, and an object column sorts alphabetically."""
    values = pd.Series(tier_values)
    return pd.Series(
        pd.Categorical(values.astype(str), categories=present, ordered=True),
        index=values.index,
        name="tier",
    )


def _origin_attribute_lookup(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """The origin-to-attributes map, deduped, indexed by origin date. Checked
    here, not later: a duplicated `ds` fans the join out before anything can look
    at it."""
    require_columns(calendar_df, ["ds", *_ORIGIN_ATTRIBUTES], "calendar_df")
    lookup = calendar_df[["ds", *_ORIGIN_ATTRIBUTES]].drop_duplicates()

    if not lookup["ds"].is_unique:
        conflicting = lookup.loc[lookup["ds"].duplicated(), "ds"].unique()
        raise ValueError(
            f"calendar maps {len(conflicting)} ds value(s) to more than one "
            f"origin attribute set; first few: {sorted(conflicting)[:5]}"
        )

    lookup = lookup.assign(ds=lookup["ds"].astype(f"datetime64[{ORIGIN_TIME_UNIT}]"))
    return lookup.set_index("ds")


def _disagreeing(left: pd.Series, right: pd.Series) -> pd.Series:
    """Rows where the two sides differ. Both-null counts as agreement: `!=` reads
    a pair of nulls as a mismatch and would send an operator upstream."""
    return (left != right) & ~(left.isna() & right.isna())


def _check_base_frame(base: pd.DataFrame) -> None:
    """The checks on the joined frame, before its `_ch`/`_bm` collapse.

    Raises:
        ValueError: On any failed check; each message names its own cause.
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
    # one config, so a disagreement is an upstream fault.
    for column in ("tier", "series_weight", "actual_monthly_total"):
        mismatched = _disagreeing(base[f"{column}_ch"], base[f"{column}_bm"])
        if mismatched.any():
            sample = base.loc[mismatched, _JOIN_KEYS].head(5)
            raise ValueError(
                f"{int(mismatched.sum())} row(s) disagree on {column!r} across the "
                f"two sidecars, which both derive it from the same panel:\n"
                f"{sample.to_string(index=False)}"
            )

    # A null on both sides is not a disagreement, so the loop above passes it.
    for column in _METRIC_INPUT_COLUMNS:
        null = base[column].isna()
        if null.any():
            sample = base.loc[null, _JOIN_KEYS].head(5)
            raise ValueError(
                f"{int(null.sum())} row(s) carry a null {column!r}, which the fold "
                f"reductions would read as zero on one side of a ratio:\n"
                f"{sample.to_string(index=False)}"
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

    Outer with an indicator so the row-set assumption becomes a column plus a
    check; wide because a ratio needs both forecasts on one row; origins are
    normalized for the output contract, since a merged key takes the left's unit.
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
        benchmark,
        on=_JOIN_KEYS,
        suffixes=("_ch", "_bm"),
        how="outer",
        indicator=True,
        # A duplicate key fans the frame out while every _merge still reads "both".
        # pandas names the offending side: left is the challenger, right the benchmark.
        validate="one_to_one",
    )
    # Not the sidecar's copies: one source, and horizon must not vary within a fold.
    origins = _origin_attribute_lookup(calendar_df)
    base["origin_fiscal_year_month"] = base["forecast_origin_date"].map(
        origins["fiscal_year_month"]
    )
    base["origin_month_fraction_elapsed"] = base["forecast_origin_date"].map(
        origins["origin_month_fraction_elapsed"]
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
    """`tier` on a score table: ordered, `global` first. A dashboard reading one
    table has no other ordering source, and a string column sorts high, low,
    middle, very_high, very_low."""
    return pd.Series(
        pd.Categorical(tier_values, categories=[GLOBAL_TIER, *present], ordered=True),
        index=tier_values.index,
        name="tier",
    )


def _model_folds(
    model_views: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> Iterator[tuple[str, tuple, pd.DataFrame, pd.DataFrame]]:
    """Every (model, fold) pair, both sides already cut to that fold. Split out
    because a fold slice needs its model, so the groupby is per-model setup that
    would otherwise wedge two statements between two `for` statements."""
    for model, (challenger, benchmark) in model_views.items():
        challenger_folds = dict(tuple(challenger.groupby(_FOLD_KEYS, observed=True)))
        benchmark_folds = dict(tuple(benchmark.groupby(_FOLD_KEYS, observed=True)))
        for fold_key, challenger_fold in challenger_folds.items():
            yield model, fold_key, challenger_fold, benchmark_folds[fold_key]


def _score_folds(
    base: pd.DataFrame, *, challenger_model: str, benchmark_model: str
) -> pd.DataFrame:
    """Every metric for both models at fold grain, in long form.

    No branch by model, so the benchmark scores itself and reads 1.0, the skill
    chart's reference line. Never test that with `== 1.0`: `wrmae_per_series`
    renormalizes weights and lands 1.0 to float tolerance. `n_obs` counts rows
    available before metric-specific exclusions, not rows used.
    """
    challenger_view = _model_view(base, "monthly_forecast_ch")
    benchmark_view = _model_view(base, "monthly_forecast_bm")

    # A model in both roles is a legitimate smoke test and must score once: two
    # identical row sets under one name would double every n_obs. One name over two
    # different row sets is not, and setdefault would drop the second in silence.
    if challenger_model == benchmark_model and not base["monthly_forecast_ch"].equals(
        base["monthly_forecast_bm"]
    ):
        raise ValueError(
            f"both roles name {challenger_model!r} while the two sidecars forecast "
            "differently, so one model's rows would be dropped without an error."
        )

    model_views = {challenger_model: (challenger_view, benchmark_view)}
    model_views.setdefault(benchmark_model, (benchmark_view, benchmark_view))

    tier_slices: list[tuple[str, str | None]] = [(GLOBAL_TIER, None)] + [
        (label, label) for label in base["tier"].cat.categories
    ]
    # Horizon is a function of the origin alone, so every fold has exactly one,
    # which is also why horizon does not multiply this table's row count.
    fold_horizons = base.drop_duplicates(_FOLD_KEYS).set_index(_FOLD_KEYS)["horizon"]

    rows = []
    for model, fold_key, challenger_fold, benchmark_fold in _model_folds(model_views):
        origin, predicted_month = fold_key

        for tier_label, tier_value in tier_slices:
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


def _derive(fold_metrics: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    """`nanmean` the fold values within a coarser grain, exactly: a fold a direct
    computation would skip is a nan this drops. The three columns aggregate
    differently, so read the agg rather than the keys."""
    derived = (
        fold_metrics.groupby(group_keys, observed=True)
        .agg(
            # The second stage every metric already does.
            value=("value", "mean"),
            # A sum: folds partition the group.
            n_obs=("n_obs", "sum"),
            # What the mean actually averaged, so nans are excluded.
            n_folds_used=("value", "count"),
        )
        .reset_index()
    )
    return derived.sort_values(group_keys).reset_index(drop=True)


def _stamp_lineage(frame: pd.DataFrame, identity: TrainRunIdentity) -> pd.DataFrame:
    """The five columns every table carries. The two uris are a transitive claim:
    evaluate never opens a Feature artifact, and the manifest and calendar checks
    are what back them."""
    return frame.assign(
        train_run_id=identity.train_run_id,
        feature_run_id=identity.feature_run_id,
        git_hash=identity.git_hash,
        panel_uri=identity.panel_uri,
        calendar_uri=identity.calendar_uri,
    )


def compute_evaluate_outputs(
    *,
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    challenger_model: str,
    benchmark_model: str,
    tier_labels: tuple[str, ...],
    identity: TrainRunIdentity,
) -> EvaluateOutputs:
    """Score one run's two models and build its four tables.

    Frames in, dataclass out, no paths, so the derivation identity is testable with
    no `tmp_path` and no files. `tier_labels` rather than the whole modeling config,
    because the tier ordering is output contract and is not recoverable from the
    data, while checks on file contents belong to `evaluate_impl`.

    Args:
        challenger_ms: The challenger sidecar's monthly_series frame.
        benchmark_ms: The benchmark sidecar's monthly_series frame.
        calendar_df: The challenger sidecar's trimmed fiscal calendar.
        challenger_model: Verified against `model_roles` by `evaluate_impl`.
        benchmark_model: Same.
        tier_labels: The configured vocabulary, lowest tier first.
        identity: Stamped into every row of all four tables.

    Raises:
        ValueError: On a failed base-frame check, or a tier value set that is not
            a prefix of `tier_labels`.

    Returns:
        EvaluateOutputs: The four frames, one per output file.
    """
    base = _build_base_frame(
        challenger_ms=challenger_ms,
        benchmark_ms=benchmark_ms,
        calendar_df=calendar_df,
        challenger_model=challenger_model,
        benchmark_model=benchmark_model,
        tier_labels=tier_labels,
    )
    fold_metrics = _score_folds(
        base, challenger_model=challenger_model, benchmark_model=benchmark_model
    )

    return EvaluateOutputs(
        per_series_comparison=_stamp_lineage(base, identity),
        fold_metrics=_stamp_lineage(fold_metrics, identity),
        period_metrics=_stamp_lineage(_derive(fold_metrics, _PERIOD_KEYS), identity),
        summary_metrics=_stamp_lineage(_derive(fold_metrics, _SCORE_KEYS), identity),
    )


def _check_directories_match_roles(
    challenger_dir: Path, benchmark_dir: Path, roles: ModelRoles
) -> None:
    """Two role names read two sidecars, one name reads one. A biconditional
    because `ModelRoles` permits one model in both roles as a smoke test."""
    if (challenger_dir != benchmark_dir) != (roles.challenger != roles.benchmark):
        raise ValueError(
            f"model_roles names {roles.challenger!r} as challenger and "
            f"{roles.benchmark!r} as benchmark, against directories\n"
            f"  challenger: {challenger_dir}\n  benchmark:  {benchmark_dir}\n"
            "which is one sidecar scored twice or two sidecars under one name."
        )


def _check_sidecar_manifests(
    manifests: dict[str, dict[str, Any]],
    identity: TrainRunIdentity,
    modeling: TrainModelingConfig,
) -> None:
    """What each sidecar declares about itself, against this run's own sources.
    Read the model names off `model_roles` instead and a swap looks consistent."""
    for role, manifest in manifests.items():
        declared = manifest["model_name"]
        expected_model = getattr(modeling.model_roles, role)
        if declared != expected_model:
            raise ValueError(
                f"the {role} directory holds {declared!r}'s sidecar while "
                f"modeling.yaml names {expected_model!r} in that role: the two "
                "directories are swapped, or another run's sidecar is wired in."
            )

        lineage = manifest["lineage"]
        if lineage["train_run_id"] != identity.train_run_id:
            raise ValueError(
                f"the {role} sidecar was backtested under train_run_id "
                f"{lineage['train_run_id']!r} while this run is "
                f"{identity.train_run_id!r}: two runs' sidecars are wired together."
            )
        if lineage["feature_run_id"] != identity.feature_run_id:
            raise ValueError(
                f"the {role} sidecar reads feature_run_id "
                f"{lineage['feature_run_id']!r} against this run's "
                f"{identity.feature_run_id!r}: the models scored different actuals."
            )

        # Whole dumped blocks, so no field set is maintained here and a legitimate
        # config addition does not break the comparison.
        for block in ("tiering", "weighting"):
            expected_block = getattr(modeling, block).model_dump()
            if manifest["config"][block] != expected_block:
                raise ValueError(
                    f"the {role} sidecar's {block} disagrees with modeling.yaml:\n"
                    f"  sidecar:  {manifest['config'][block]}\n"
                    f"  modeling: {expected_block}\n"
                    "re-composing under an existing run id overwrites that file in "
                    "place, so every table would stamp settings its scores predate."
                )


def _check_calendars_agree(
    challenger_calendar: pd.DataFrame,
    benchmark_calendar: pd.DataFrame,
    forecast_origins: pd.Series,
) -> None:
    """The two sidecars map the forecast origins to the same attributes. Those
    rows and by value: nothing else here can change a number this step emits."""
    # Not normalized to ORIGIN_TIME_UNIT: reindex bridges units; index never written
    origins = pd.Index(forecast_origins.unique(), name="ds").sort_values()
    challenger_origins = _origin_attribute_lookup(challenger_calendar).reindex(origins)
    benchmark_origins = _origin_attribute_lookup(benchmark_calendar).reindex(origins)

    disagreeing = origins[
        (challenger_origins != benchmark_origins).any(axis=1).to_numpy()
    ]
    if len(disagreeing):
        raise ValueError(
            f"the two sidecars' calendars disagree on {len(disagreeing)} forecast "
            "origin(s), by value or by one of them lacking it; first few: "
            f"{[str(ds.date()) for ds in disagreeing[:5]]}. A shifted month "
            "boundary moves rows between horizons rather than raising."
        )


def _hero_metric_values(
    summary_metrics: pd.DataFrame, challenger_model: str
) -> dict[str, float | None]:
    """The challenger's global-tier hero metric per horizon, read off the table
    rather than recomputed, so the manifest cannot disagree with the file beside it."""
    hero = summary_metrics[
        (summary_metrics["model"] == challenger_model)
        & (summary_metrics["tier"] == GLOBAL_TIER)
        & (summary_metrics["metric"] == _HERO_METRIC_NAME)
    ]
    return dict(zip(hero["horizon"], hero["value"], strict=True))


def _build_summary(
    outputs: EvaluateOutputs, identity: TrainRunIdentity, roles: ModelRoles
) -> EvaluateSummary:
    """The run's headline result, counted off the tables just built.
    `n_folds_total` counts every fold event, not any cell's contributing folds."""
    base = outputs.per_series_comparison
    origins = [
        pd.Timestamp(origin).date().isoformat()
        for origin in sorted(base["forecast_origin_date"].unique())
    ]
    return EvaluateSummary(
        train_run_id=identity.train_run_id,
        challenger_model=roles.challenger,
        benchmark_model=roles.benchmark,
        feature_run_id=identity.feature_run_id,
        n_origins=len(origins),
        first_origin=origins[0],
        last_origin=origins[-1],
        n_series=int(base["unique_id"].nunique()),
        n_folds_total=len(base[_FOLD_KEYS].drop_duplicates()),
        hero_metric_name=_HERO_METRIC_NAME,
        hero_metric_values=_hero_metric_values(
            outputs.summary_metrics, roles.challenger
        ),
        output_rows={
            filename: len(getattr(outputs, field))
            for field, filename in _OUTPUT_FILENAMES.items()
        },
    )


def _build_manifest(
    summary: EvaluateSummary,
    identity: TrainRunIdentity,
    modeling: TrainModelingConfig,
) -> dict[str, Any]:
    """Explicit mapping from three single sources, so nothing here is counted twice.
    The config blocks are restated because `evaluate/` holds no sidecar to point at."""
    return {
        "challenger_model": summary.challenger_model,
        "benchmark_model": summary.benchmark_model,
        "lineage": {
            "train_run_id": identity.train_run_id,
            "feature_run_id": identity.feature_run_id,
            "git_hash": identity.git_hash,
            "panel_uri": identity.panel_uri,
            "calendar_uri": identity.calendar_uri,
        },
        "config": {
            "tiering": modeling.tiering.model_dump(),
            "weighting": modeling.weighting.model_dump(),
        },
        "hero_metric": {
            "name": summary.hero_metric_name,
            "tier": GLOBAL_TIER,
            "values": _json_floats(summary.hero_metric_values),
        },
        "origins": {
            "n_origins": summary.n_origins,
            "first_origin": summary.first_origin,
            "last_origin": summary.last_origin,
        },
        "n_series": summary.n_series,
        "n_folds_total": summary.n_folds_total,
        "output_rows": summary.output_rows,
    }


def evaluate_impl(
    *,
    challenger_dir: Path,
    benchmark_dir: Path,
    compose_configs_dir: Path,
    out_dir: Path,
) -> EvaluateSummary:
    """Score one run's challenger against its benchmark and write its four tables.

    Keyword-only, and the hazard is worse than backtest's: a transposed pair never
    surfaces, since inverted skill ratios are a complete and plausible table. No
    model names and no provenance scalars, both read instead from sources the
    wrapper does not mediate, because a guard is only as good as the independence
    of its two sides. The completion marker is deleted before the first frame is
    read, so a failed rerun leaves none over a half-rewritten directory.

    Args:
        challenger_dir: The challenger's backtest sidecar.
        benchmark_dir: The benchmark's, which is the same directory when one
            model holds both roles.
        compose_configs_dir: Holds `modeling.yaml`, with `run_identity.json`
            beside it. Not derivable from `out_dir`: the run-root check needs two
            independently supplied values or it compares `out_dir` to itself.
        out_dir: This run's evaluate directory, created if missing.

    Raises:
        ValueError: If `out_dir` is not this run's evaluate directory, if
            `run_identity.json` is absent, or on a failed role, lineage, config,
            calendar or base-frame check.
        ValidationError: If the identity or `modeling.yaml` fails to revalidate.

    Returns:
        EvaluateSummary: The run's headline result, for a wrapper or the runner.
    """
    if out_dir.name != _STEP_DIR_NAME:
        raise ValueError(
            f"out_dir {out_dir} must be named {_STEP_DIR_NAME!r}: a sibling step "
            "directory satisfies the run-root check and fails only this one."
        )

    identity_path = compose_configs_dir.parent / "run_identity.json"
    if not identity_path.is_file():
        raise ValueError(
            f"No run_identity.json at {identity_path}: compose_configs writes it "
            "beside its step directory, so that path is wrong or the step failed."
        )
    identity = TrainRunIdentity.model_validate_json(identity_path.read_text())

    # Against the parsed id, not compose_configs_dir.parent as paths: the two arrive by
    # different channels, so two spellings of one place would fire on correct wiring.
    if out_dir.parent.name != identity.train_run_id:
        raise ValueError(
            f"out_dir {out_dir} must sit under the run root "
            f"{identity.train_run_id!r}: every row of four tables would otherwise "
            "state a run the tables are not filed under."
        )

    modeling = TrainModelingConfig.model_validate(
        yaml.safe_load((compose_configs_dir / "modeling.yaml").read_text())
    )
    roles = modeling.model_roles
    _check_directories_match_roles(challenger_dir, benchmark_dir, roles)

    manifests = {
        "challenger": json.loads((challenger_dir / _SIDECAR_MANIFEST).read_text()),
        "benchmark": json.loads((benchmark_dir / _SIDECAR_MANIFEST).read_text()),
    }
    _check_sidecar_manifests(manifests, identity, modeling)

    # Otherwise a failed rerun leaves the old manifest over a mix of two runs' tables.
    (out_dir / _MANIFEST_FILENAME).unlink(missing_ok=True)

    # Every frame is read below the unlink, so an unreadable one leaves no marker.
    challenger_ms = pd.read_parquet(challenger_dir / _SIDECAR_MONTHLY_SERIES)
    benchmark_ms = pd.read_parquet(benchmark_dir / _SIDECAR_MONTHLY_SERIES)
    # The challenger's calendar is the one used, the challenger being the subject.
    calendar_df = pd.read_parquet(challenger_dir / _SIDECAR_CALENDAR)
    _check_calendars_agree(
        calendar_df,
        pd.read_parquet(benchmark_dir / _SIDECAR_CALENDAR),
        challenger_ms["forecast_origin_date"],
    )

    outputs = compute_evaluate_outputs(
        challenger_ms=challenger_ms,
        benchmark_ms=benchmark_ms,
        calendar_df=calendar_df,
        challenger_model=roles.challenger,
        benchmark_model=roles.benchmark,
        tier_labels=tuple(modeling.tiering.tier_labels),
        identity=identity,
    )
    summary = _build_summary(outputs, identity, roles)

    out_dir.mkdir(parents=True, exist_ok=True)
    for field, filename in _OUTPUT_FILENAMES.items():
        getattr(outputs, field).to_parquet(out_dir / filename, index=False)

    manifest = _build_manifest(summary, identity, modeling)
    # evaluate_manifest.json written is the step's completion marker. Keep this last.
    (out_dir / _MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")

    _log.info(
        "evaluate complete: challenger=%s benchmark=%s folds=%d %s=%s out_dir=%s",
        summary.challenger_model,
        summary.benchmark_model,
        summary.n_folds_total,
        summary.hero_metric_name,
        summary.hero_metric_values,
        out_dir,
    )
    return summary
