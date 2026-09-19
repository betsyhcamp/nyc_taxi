import json
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import yaml

from fcstnyctaxi.core.train.evaluate_impl import (
    _FOLD_KEYS,
    _PERIOD_KEYS,
    _RELATIVE_METRIC_FNS,
    _SCORE_KEYS,
    GLOBAL_TIER,
    EvaluateOutputs,
    _build_base_frame,
    _check_base_frame,
    _model_view,
    _score_folds,
    compute_evaluate_outputs,
)
from fcstnyctaxi.lib.config.bindings import (
    model_names_from_roles,
    train_modeling_bindings,
)
from fcstnyctaxi.lib.config.composition import compose_config, save_config
from fcstnyctaxi.lib.fold_metrics import compute_wape, compute_wrmae_pooled
from fcstnyctaxi.lib.period_utils import ORIGIN_TIME_UNIT, derive_horizon_label
from fcstnyctaxi.lib.storage_layout import composed_config_filename
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.config.train import TrainModelingConfig
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity

CONFIG_DIR = get_project_root_dir() / "config"
FEATURE_RUN_ID = "f-2026-09-18"
TRAIN_RUN_ID = "t-2026-09-18"

# ================================================
# The fixture: two synthetic backtest sidecars and the run root above them
#
# Inputs only, so nothing here anticipates evaluate's output shape.
# 24 weeks, W-SUN, four weeks a fiscal month.
#
#   origin        fiscal month  fraction  forecasts
#   2024-02-11    202402        0.50      202402 h1, 202403 h2
#   2024-02-25    202402        1.00      202403 h1, 202404 h2
#   2024-03-10    202403        0.50      202403 h1, 202404 h2
#
# 202403 is reached at both horizons, the month-end origin exercises
# derive_horizon_label's frac == 1.0 branch, and the predicted months carry 1, 3
# and 2 origins, which is what stops summary being a mean of period values.
# ================================================

_WEEKS_PER_MONTH = 4
_N_WEEKS = 24
_WEEKS = pd.date_range("2024-01-07", periods=_N_WEEKS, freq="W-SUN")
_MONTHS = [202401 + month for month in range(_N_WEEKS // _WEEKS_PER_MONTH)]

# (week index, the months that origin forecasts). Stated rather than derived, so
# a reader sees the expected horizons; a self-check confirms the labeller agrees.
_ORIGIN_TARGETS = (
    (5, (202402, 202403)),
    (7, (202403, 202404)),
    (9, (202403, 202404)),
)

_SERIES = ("time_series_a", "time_series_b", "time_series_c")

# Index into the present tier labels, per series and origin:
#
#   origin 0  2 bins:  a=low     b=very_low  c=very_low
#   origin 1  3 bins:  a=middle  b=low       c=very_low
#   origin 2  3 bins:  a=middle  b=very_low  c=low
#
# One fact drives it. time_series_c is dormant at the first origin, so
# assign_tiers bins two series there and force-assigns it tier_labels[0], then
# bins three once it recovers. Nothing here is stipulated: the top tier has no
# series at origin 0, which is the empty slice a metric returns nan for; the
# category sets therefore differ by origin, which degrades a concatenated
# categorical to object; and two series migrate for two reasons, a because the
# bin count moved and b because c's mean of positive weeks overtook it.
#
# Three and not two: qcut spreads across every bin it makes, so
# min(len(tier_labels), mean_pos.nunique()) is 3 whenever all three are active.
_TIER_INDEX = {
    "time_series_a": (1, 2, 2),
    "time_series_b": (0, 1, 0),
    "time_series_c": (0, 0, 1),
}

# cbrt of a 26-week trailing sum, clipped at zero, so it moves with the origin.
# Series c carries none while dormant, the cell the per-series reductions used to
# score as a perfect forecast. It stays under b's even after its tier passes it:
# tier averages the positive weeks alone, weight the whole window.
_WEIGHT = {
    "time_series_a": (29.6, 29.6, 29.6),
    "time_series_b": (17.3, 17.3, 17.3),
    "time_series_c": (0.0, 6.7, 10.3),
}

# What each series realizes: a fact about the month, not the origin. Series c is
# positive throughout, since tier and weight look back while the actual looks
# forward, so a dormant series still has revenue to forecast.
_ACTUAL_LEVEL = {"time_series_a": 1000.0, "time_series_b": 200.0, "time_series_c": 50.0}


def _shipped_modeling() -> TrainModelingConfig:
    """The committed tiering, weighting and model roles, composed not hand-built."""
    return cast(
        TrainModelingConfig,
        compose_config(CONFIG_DIR, train_modeling_bindings()).config,
    )


def _present_tier_labels(modeling: TrainModelingConfig) -> tuple[str, ...]:
    """The labels this data uses, a proper prefix of the configured five.

    A proper prefix is the point: a run bins as many tiers as its data supports,
    so anything sourcing the vocabulary from config emits rows for absent tiers.
    """
    return tuple(modeling.tiering.tier_labels[:3])


def _identity() -> TrainRunIdentity:
    """The provenance record compose_configs leaves at the run root."""
    return TrainRunIdentity(
        git_hash="abc1234-dirty",
        feature_run_id=FEATURE_RUN_ID,
        train_run_id=TRAIN_RUN_ID,
        panel_uri=f"gs://bucket/dev/feature/{FEATURE_RUN_ID}/time_series.parquet",
        calendar_uri=f"gs://bucket/dev/feature/{FEATURE_RUN_ID}/fiscal_calendar.parquet",
    )


def _calendar() -> pd.DataFrame:
    """Every column the contract declares, on a unit the origins do not share.

    The calendar is a trimmed input snapshot carrying whatever unit Feature
    wrote, while backtest normalizes its outputs to ORIGIN_TIME_UNIT. Agreeing
    units would leave the normalization unexercised.
    """
    month_index = [week // _WEEKS_PER_MONTH for week in range(_N_WEEKS)]
    week_of_month = [week % _WEEKS_PER_MONTH + 1 for week in range(_N_WEEKS)]
    return pd.DataFrame(
        {
            "ds": _WEEKS.astype("datetime64[us]"),
            "fiscal_year_month": [_MONTHS[month] for month in month_index],
            "fiscal_month": [month % 12 + 1 for month in month_index],
            "fiscal_week_of_month": week_of_month,
            "weeks_in_month": _WEEKS_PER_MONTH,
            "count_workdays": 5,
            "origin_month_fraction_elapsed": [
                week / _WEEKS_PER_MONTH for week in week_of_month
            ],
            "fiscal_year": [_MONTHS[month] // 100 for month in month_index],
            "fiscal_year_week": list(range(1, _N_WEEKS + 1)),
        }
    )


def _folds() -> list[tuple[int, int, int]]:
    """(origin index, week index, predicted fiscal month), one per fold, in order."""
    return [
        (origin_index, week, month)
        for origin_index, (week, months) in enumerate(_ORIGIN_TARGETS)
        for month in months
    ]


def _actual(series: str, month: int) -> float:
    """Grows 5% a month from 202402. Never varies by origin."""
    return _ACTUAL_LEVEL[series] * (1 + 0.05 * (month - 202402))


def _relative_error(
    model: str, fold_index: int, series_index: int, *, benchmark: str
) -> float:
    """Signed relative forecast error, varying by fold and by series.

    By series matters: a constant-within-fold error makes the pooled and
    per-series metrics equal, so a bug swapping them would pass. The challenger
    stays under the benchmark everywhere and turns negative for the smaller
    series, so it wins at every slice and signed bias carries both signs.
    """
    if model == benchmark:
        return 0.20 + 0.03 * fold_index + 0.02 * series_index
    return 0.12 - 0.02 * fold_index - 0.09 * series_index


def _monthly_series(
    model: str,
    modeling: TrainModelingConfig,
    *,
    tier_as_string: bool = False,
) -> pd.DataFrame:
    """One sidecar's monthly_series frame, the eight columns backtest writes.

    tier is an ordered categorical built directly rather than through
    assign_tiers, whose degenerate early return would have the fixture exercising
    that function's edge cases instead of evaluate's. tier_as_string is the other
    legal shape, and the one this data would really produce: the first origin
    bins fewer tiers, so a concatenated categorical degrades to object.
    """
    labels = _present_tier_labels(modeling)
    calendar = _calendar().set_index("ds")

    rows = []
    for fold_index, (origin_index, week, month) in enumerate(_folds()):
        origin = _WEEKS[week]
        for series_index, series in enumerate(_SERIES):
            actual = _actual(series, month)
            error = _relative_error(
                model,
                fold_index,
                series_index,
                benchmark=modeling.model_roles.benchmark,
            )
            rows.append(
                {
                    "forecast_origin_date": origin,
                    "predicted_fiscal_year_month": month,
                    "unique_id": series,
                    "tier": labels[_TIER_INDEX[series][origin_index]],
                    "monthly_forecast": actual * (1 + error),
                    "actual_monthly_total": actual,
                    "series_weight": _WEIGHT[series][origin_index],
                    "origin_month_fraction_elapsed": calendar.loc[
                        origin.as_unit("us"), "origin_month_fraction_elapsed"
                    ],
                }
            )

    frame = pd.DataFrame(rows)
    frame["forecast_origin_date"] = frame["forecast_origin_date"].astype(
        f"datetime64[{ORIGIN_TIME_UNIT}]"
    )
    if not tier_as_string:
        frame["tier"] = pd.Categorical(frame["tier"], categories=labels, ordered=True)
    return frame


def _manifest(
    model: str,
    monthly_series: pd.DataFrame,
    modeling: TrainModelingConfig,
    identity: TrainRunIdentity,
) -> dict:
    """backtest_manifest.json, the shape backtest's _build_manifest emits."""
    origins = sorted(monthly_series["forecast_origin_date"].unique())
    return {
        "model_name": model,
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
        "origins": {
            "n_origins": len(origins),
            "first_origin": pd.Timestamp(origins[0]).date().isoformat(),
            "last_origin": pd.Timestamp(origins[-1]).date().isoformat(),
        },
        "n_series": int(monthly_series["unique_id"].nunique()),
        "output_rows": {"monthly_series.parquet": len(monthly_series)},
    }


def _stage_sidecar(
    sidecar_dir: Path,
    model: str,
    modeling: TrainModelingConfig,
    identity: TrainRunIdentity,
    *,
    tier_as_string: bool = False,
) -> None:
    """Write one model's sidecar: the three files evaluate reads, plus the config.

    Evaluate never opens composed_config.yaml. It is written anyway so the
    fixture is a whole sidecar, and as a placeholder because nothing validates it.
    """
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    monthly_series = _monthly_series(model, modeling, tier_as_string=tier_as_string)

    monthly_series.to_parquet(sidecar_dir / "monthly_series.parquet", index=False)
    _calendar().to_parquet(sidecar_dir / "fiscal_calendar.parquet", index=False)
    (sidecar_dir / composed_config_filename(model)).write_text(
        f"model:\n  fit_predict_callable: {model}\n"
    )
    (sidecar_dir / "backtest_manifest.json").write_text(
        json.dumps(_manifest(model, monthly_series, modeling, identity), indent=2)
        + "\n"
    )


def stage_run(
    root: Path,
    modeling: TrainModelingConfig,
    *,
    identity: TrainRunIdentity | None = None,
    tier_as_string: bool = False,
) -> dict[str, Path]:
    """A whole run root, and evaluate_impl's four arguments naming its parts.

    The sidecar set comes from model_names_from_roles, so a config naming one
    model in both roles stages one directory and both roles point at it.
    """
    identity = identity or _identity()
    run_root = root / identity.train_run_id
    compose_configs_dir = run_root / "compose_configs"
    compose_configs_dir.mkdir(parents=True, exist_ok=True)

    (run_root / "run_identity.json").write_text(identity.model_dump_json(indent=2))
    save_config(
        modeling.model_dump(by_alias=True, exclude_none=True),
        compose_configs_dir / "modeling.yaml",
    )

    roles = modeling.model_roles
    for model in model_names_from_roles(roles):
        _stage_sidecar(
            run_root / "backtest" / model,
            model,
            modeling,
            identity,
            tier_as_string=tier_as_string,
        )

    return {
        "challenger_dir": run_root / "backtest" / roles.challenger,
        "benchmark_dir": run_root / "backtest" / roles.benchmark,
        "compose_configs_dir": compose_configs_dir,
        "out_dir": run_root / "evaluate",
    }


@pytest.fixture(scope="module")
def modeling() -> TrainModelingConfig:
    """The shipped tiering, weighting and roles: naive against lightgbm."""
    return _shipped_modeling()


@pytest.fixture(scope="module")
def calendar_df() -> pd.DataFrame:
    """The sidecar calendar both models carry."""
    return _calendar()


@pytest.fixture(scope="module")
def challenger_ms(modeling: TrainModelingConfig) -> pd.DataFrame:
    """The challenger's monthly_series, for the pure function's no-files tests."""
    return _monthly_series(modeling.model_roles.challenger, modeling)


@pytest.fixture(scope="module")
def benchmark_ms(modeling: TrainModelingConfig) -> pd.DataFrame:
    """The benchmark's monthly_series, same grain and keys as the challenger's."""
    return _monthly_series(modeling.model_roles.benchmark, modeling)


@pytest.fixture
def staged_run(tmp_path: Path, modeling: TrainModelingConfig) -> dict[str, Path]:
    """A staged run root, per test, for anything that reads or writes files."""
    return stage_run(tmp_path, modeling)


# ================================================
# The fixture's own checks
#
# Without them a broken fixture reads as a broken check. Each asserts one hazard
# the fixture claims to carry, using library code rather than evaluate's.
# ================================================


def _horizons(monthly_series: pd.DataFrame, calendar: pd.DataFrame) -> pd.Series:
    """Horizon labels for a monthly_series frame, the way evaluate will derive them."""
    origin_month = monthly_series["forecast_origin_date"].map(
        calendar.assign(ds=calendar["ds"].astype(f"datetime64[{ORIGIN_TIME_UNIT}]"))
        .drop_duplicates("ds")
        .set_index("ds")["fiscal_year_month"]
    )
    return derive_horizon_label(
        predicted_fiscal_year_month=monthly_series["predicted_fiscal_year_month"],
        origin_fiscal_year_month=origin_month,
        origin_month_fraction_elapsed=monthly_series["origin_month_fraction_elapsed"],
    )


def test_the_fixture_reaches_one_month_at_two_horizons(
    challenger_ms: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """A fixture whose months each have one horizon cannot separate the two axes."""
    labelled = challenger_ms.assign(horizon=_horizons(challenger_ms, calendar_df))

    per_month = labelled.groupby("predicted_fiscal_year_month")["horizon"].nunique()
    assert set(labelled["horizon"]) == {"horizon_1", "horizon_2"}
    assert (per_month > 1).any()


def test_the_month_end_origin_shifts_its_own_horizons(
    challenger_ms: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Getting the frac == 1.0 branch wrong moves labels rather than raising."""
    labelled = challenger_ms.assign(horizon=_horizons(challenger_ms, calendar_df))
    month_end = labelled[labelled["origin_month_fraction_elapsed"] == 1.0]

    assert not month_end.empty
    assert (labelled["origin_month_fraction_elapsed"] < 1.0).any()
    # Its own fiscal month counts as completed, so it forecasts the next one first.
    assert month_end["predicted_fiscal_year_month"].min() == 202403


def test_predicted_months_carry_unequal_origin_counts(
    challenger_ms: pd.DataFrame,
) -> None:
    """The condition that makes summary unrecoverable from period: with equal
    counts a mean of period values would equal a mean of fold values."""
    per_month = challenger_ms.groupby("predicted_fiscal_year_month")[
        "forecast_origin_date"
    ].nunique()

    assert per_month.nunique() > 1


def test_a_series_changes_tier_between_origins(challenger_ms: pd.DataFrame) -> None:
    """assign_tiers re-bins at every origin, so tier belongs to no per-series grain."""
    per_series = challenger_ms.groupby("unique_id", observed=True)["tier"].nunique()

    assert (per_series > 1).any()


def test_the_origins_do_not_all_bin_the_same_number_of_tiers(
    challenger_ms: pd.DataFrame, modeling: TrainModelingConfig
) -> None:
    """Two hazards in one fact: an absent tier makes that fold's slice empty, the
    nan a nanmean drops, and differing category sets degrade a categorical."""
    per_origin = challenger_ms.groupby("forecast_origin_date", observed=True)[
        "tier"
    ].nunique()

    assert per_origin.nunique() > 1
    assert set(challenger_ms["tier"].astype(str)) == set(_present_tier_labels(modeling))


def test_a_dormant_series_carries_no_weight_at_that_origin(
    challenger_ms: pd.DataFrame,
) -> None:
    """The cell the per-series reductions used to score 0.0, the best attainable
    value. All-positive weights would never reach the path that fix opened."""
    per_origin = challenger_ms.groupby("forecast_origin_date", observed=True)[
        "series_weight"
    ].min()

    assert (per_origin == 0).any()
    assert (per_origin > 0).any()


def test_the_tier_values_are_a_proper_prefix_of_the_configured_labels(
    challenger_ms: pd.DataFrame, modeling: TrainModelingConfig
) -> None:
    """Sourcing tiers from config would emit rows for tiers no series has, which
    a fixture using every configured label could not show."""
    configured = list(modeling.tiering.tier_labels)
    present = set(challenger_ms["tier"].astype(str))

    assert present == set(configured[: len(present)])
    assert len(present) < len(configured)


def test_the_two_sidecars_agree_on_everything_but_the_forecast(
    challenger_ms: pd.DataFrame, benchmark_ms: pd.DataFrame
) -> None:
    """Both derive from one panel under one config, so a disagreement is an
    upstream fault. The forecasts must differ, or the comparison is degenerate."""
    keys = ["forecast_origin_date", "predicted_fiscal_year_month", "unique_id"]
    shared = ["tier", "series_weight", "actual_monthly_total"]

    challenger = challenger_ms.sort_values(keys).reset_index(drop=True)
    benchmark = benchmark_ms.sort_values(keys).reset_index(drop=True)

    pd.testing.assert_frame_equal(challenger[keys + shared], benchmark[keys + shared])
    assert (challenger["monthly_forecast"] != benchmark["monthly_forecast"]).all()


def test_the_challenger_is_closer_than_the_benchmark_everywhere(
    challenger_ms: pd.DataFrame, benchmark_ms: pd.DataFrame
) -> None:
    """Skill below 1.0 at every slice, so a ratio inversion cannot hide here."""
    challenger_error = (
        challenger_ms["monthly_forecast"] - challenger_ms["actual_monthly_total"]
    ).abs()
    benchmark_error = (
        benchmark_ms["monthly_forecast"] - benchmark_ms["actual_monthly_total"]
    ).abs()

    assert (challenger_error < benchmark_error).all()


def test_the_challenger_over_and_under_forecasts(challenger_ms: pd.DataFrame) -> None:
    """One sign everywhere cannot distinguish a signed metric from an absolute."""
    signed = challenger_ms["monthly_forecast"] - challenger_ms["actual_monthly_total"]

    assert (signed > 0).any()
    assert (signed < 0).any()


def test_the_actual_is_a_fact_about_the_month_not_the_origin(
    challenger_ms: pd.DataFrame,
) -> None:
    """The upstream property every metric depends on, and the passing case the
    guard that checks it needs."""
    per_series_month = challenger_ms.groupby(
        ["unique_id", "predicted_fiscal_year_month"], observed=True
    )["actual_monthly_total"].nunique()

    assert (per_series_month == 1).all()


def test_the_calendar_and_the_origins_disagree_on_time_unit(
    challenger_ms: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """The normalization needs work to do. pandas bridges the units on the way
    in, so what a matching fixture would hide is the frame carrying the sidecar's
    unit rather than this project's."""
    assert calendar_df["ds"].dt.unit != challenger_ms["forecast_origin_date"].dt.unit


def test_a_staged_sidecar_round_trips_its_tier_and_its_calendar_unit(
    staged_run: dict[str, Path], challenger_ms: pd.DataFrame
) -> None:
    """Parquet sits between the fixture and every file-reading test, so a dtype
    it drops is a property those tests do not actually have."""
    written = pd.read_parquet(staged_run["challenger_dir"] / "monthly_series.parquet")
    calendar = pd.read_parquet(staged_run["challenger_dir"] / "fiscal_calendar.parquet")

    pd.testing.assert_frame_equal(written, challenger_ms)
    assert written["tier"].cat.ordered
    assert calendar["ds"].dt.unit != written["forecast_origin_date"].dt.unit


def test_the_string_tier_sidecar_carries_no_categorical(
    tmp_path: Path, modeling: TrainModelingConfig
) -> None:
    """A valid sidecar can hold a plain string tier, which this fixture's uneven
    bin counts would really produce. Without it the coercion is untested and
    every categorical assertion passes for the wrong reason."""
    paths = stage_run(tmp_path, modeling, tier_as_string=True)

    written = pd.read_parquet(paths["challenger_dir"] / "monthly_series.parquet")

    assert written["tier"].dtype == object
    assert set(written["tier"]) == set(_present_tier_labels(modeling))


def test_the_same_model_run_stages_one_sidecar_for_both_roles(
    tmp_path: Path, modeling: TrainModelingConfig
) -> None:
    """ModelRoles permits one name in both roles as a smoke test, which only
    works if both roles read one sidecar."""
    same = modeling.model_copy(
        update={
            "model_roles": modeling.model_roles.model_copy(
                update={"benchmark": modeling.model_roles.challenger}
            )
        }
    )

    paths = stage_run(tmp_path, same)

    assert paths["challenger_dir"] == paths["benchmark_dir"]
    assert paths["challenger_dir"].is_dir()


def test_the_staged_manifests_name_their_own_model_and_one_run(
    staged_run: dict[str, Path], modeling: TrainModelingConfig
) -> None:
    """Each sidecar names its own model, which is what catches a swap: read the
    names off the roles instead and a swapped pair labels itself consistently."""
    identity = json.loads(
        (staged_run["compose_configs_dir"].parent / "run_identity.json").read_text()
    )

    for role in ("challenger", "benchmark"):
        manifest = json.loads(
            (staged_run[f"{role}_dir"] / "backtest_manifest.json").read_text()
        )
        assert manifest["model_name"] == getattr(modeling.model_roles, role)
        assert manifest["lineage"]["train_run_id"] == identity["train_run_id"]
        assert manifest["lineage"]["feature_run_id"] == identity["feature_run_id"]
        assert manifest["config"]["tiering"] == modeling.tiering.model_dump()
        assert manifest["config"]["weighting"] == modeling.weighting.model_dump()


def test_the_staged_modeling_yaml_revalidates_to_what_was_written(
    staged_run: dict[str, Path], modeling: TrainModelingConfig
) -> None:
    """Evaluate reads the roles from this file, so a yaml that does not round
    trip would fail guards on the fixture's own writing."""
    reloaded = TrainModelingConfig.model_validate(
        yaml.safe_load(
            (staged_run["compose_configs_dir"] / "modeling.yaml").read_text()
        )
    )

    assert reloaded.model_roles == modeling.model_roles
    assert reloaded.tiering == modeling.tiering
    assert reloaded.weighting == modeling.weighting


# ================================================
# _build_base_frame and its guards
#
# The fixtures are module-scoped, so every break below works on a copy.
# ================================================


def _base(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
    **overrides: object,
) -> pd.DataFrame:
    """_build_base_frame on the fixture, any argument replaceable by name."""
    kwargs: dict[str, object] = {
        "challenger_ms": challenger_ms,
        "benchmark_ms": benchmark_ms,
        "calendar_df": calendar_df,
        "challenger_model": modeling.model_roles.challenger,
        "benchmark_model": modeling.model_roles.benchmark,
        "tier_labels": tuple(modeling.tiering.tier_labels),
    }
    return _build_base_frame(**{**kwargs, **overrides})


def _changed(frame: pd.DataFrame, column: str, value: object) -> pd.DataFrame:
    """A copy with one cell replaced, for a one-way break."""
    changed = frame.copy()
    changed.loc[changed.index[0], column] = value
    return changed


def test_the_base_frame_keeps_one_row_per_origin_month_and_series(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
) -> None:
    """A fanned join inflates every weighted sum while the run looks healthy."""
    base = _base(challenger_ms, benchmark_ms, calendar_df, modeling)

    assert not base.duplicated(
        ["forecast_origin_date", "predicted_fiscal_year_month", "unique_id"]
    ).any()
    assert (base["_merge"] == "both").all()
    assert len(base) == len(challenger_ms)


def test_the_base_frame_labels_horizons_off_the_calendar(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
) -> None:
    """A horizon off the wrong origin month is a wrong number, not a crash."""
    base = _base(challenger_ms, benchmark_ms, calendar_df, modeling)

    expected = challenger_ms.assign(horizon=_horizons(challenger_ms, calendar_df))
    merged = base.merge(
        expected[["forecast_origin_date", "predicted_fiscal_year_month", "horizon"]],
        on=["forecast_origin_date", "predicted_fiscal_year_month"],
        suffixes=("", "_expected"),
    )
    assert (merged["horizon"] == merged["horizon_expected"]).all()


def test_the_base_frame_names_the_model_behind_each_forecast(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
) -> None:
    """The frame is wide, so nothing else in it says which model _ch is."""
    base = _base(challenger_ms, benchmark_ms, calendar_df, modeling)

    assert set(base["challenger_model"]) == {modeling.model_roles.challenger}
    assert set(base["benchmark_model"]) == {modeling.model_roles.benchmark}
    assert (base["monthly_forecast_ch"] != base["monthly_forecast_bm"]).all()


def test_the_origin_unit_is_normalized_whatever_the_sidecar_carries(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
) -> None:
    """pandas bridges the units, so this protects the output contract and not the
    join: a merged key inherits the left side's unit."""
    as_micros = challenger_ms.assign(
        forecast_origin_date=challenger_ms["forecast_origin_date"].astype(
            "datetime64[us]"
        )
    )

    base = _base(as_micros, benchmark_ms, calendar_df, modeling)

    assert (base["_merge"] == "both").all()
    assert base["forecast_origin_date"].dt.unit == ORIGIN_TIME_UNIT


def test_keys_that_look_equal_but_are_not_are_named_as_one_sided(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
) -> None:
    """The only silent case: pandas bridges units and raises on a dtype family
    mismatch, so only two same-dtype keys that differ one-side a join."""
    shifted = challenger_ms.assign(
        forecast_origin_date=challenger_ms["forecast_origin_date"]
        + pd.Timedelta(hours=12)
    )

    with pytest.raises(ValueError, match="matched one side only"):
        _base(shifted, benchmark_ms, calendar_df, modeling)


@pytest.mark.parametrize("dropped_from", ["challenger_ms", "benchmark_ms"])
def test_a_row_only_one_side_carries_is_named_by_direction(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
    dropped_from: str,
) -> None:
    """The two directions have different owners, so a bare count misdirects."""
    frames = {"challenger_ms": challenger_ms, "benchmark_ms": benchmark_ms}
    frames[dropped_from] = frames[dropped_from].iloc[1:]
    expected = "right_only" if dropped_from == "challenger_ms" else "left_only"

    with pytest.raises(ValueError, match=expected):
        _base(calendar_df=calendar_df, modeling=modeling, **frames)


@pytest.mark.parametrize(
    ("column", "value"),
    [("series_weight", 99.0), ("actual_monthly_total", 1.0), ("tier", "middle")],
)
def test_the_two_sides_disagreeing_on_a_shared_column_raises(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
    column: str,
    value: object,
) -> None:
    """All three come off one panel under one config, so a disagreement is
    upstream rather than a finding about either model."""
    with pytest.raises(ValueError, match=f"disagree on {column!r}"):
        _base(
            challenger_ms,
            _changed(benchmark_ms, column, value),
            calendar_df,
            modeling,
        )


def test_an_actual_that_varies_by_origin_raises(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
) -> None:
    """A realized total cannot depend on its origin. The month must be one several
    origins reach, and both sides must move or the agreement check fires first."""
    reached = challenger_ms.groupby(
        ["unique_id", "predicted_fiscal_year_month"], observed=True
    )["forecast_origin_date"].nunique()
    series, month = reached[reached > 1].index[0]
    series_month = (challenger_ms["unique_id"] == series) & (
        challenger_ms["predicted_fiscal_year_month"] == month
    )
    origin = challenger_ms.loc[series_month, "forecast_origin_date"].iloc[0]

    def _bend_one_origin(frame: pd.DataFrame) -> pd.DataFrame:
        bent = frame.copy()
        bent.loc[
            (bent["unique_id"] == series)
            & (bent["predicted_fiscal_year_month"] == month)
            & (bent["forecast_origin_date"] == origin),
            "actual_monthly_total",
        ] = 1.0
        return bent

    with pytest.raises(ValueError, match="more than one\\s+actual_monthly_total"):
        _base(
            _bend_one_origin(challenger_ms),
            _bend_one_origin(benchmark_ms),
            calendar_df,
            modeling,
        )


def test_a_calendar_mapping_one_week_to_two_months_raises(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
) -> None:
    """Must fire before the join: a duplicated ds fans the frame out."""
    conflicting = calendar_df.head(1).assign(fiscal_year_month=209912)
    fanned = pd.concat([calendar_df, conflicting], ignore_index=True)

    with pytest.raises(ValueError, match="more than one"):
        _base(challenger_ms, benchmark_ms, fanned, modeling)


def test_an_origin_absent_from_the_calendar_raises(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
) -> None:
    """derive_horizon_label emits horizon_nan on a null rather than raising."""
    origin = challenger_ms["forecast_origin_date"].iloc[0]
    trimmed = calendar_df[calendar_df["ds"] != origin.as_unit("us")]

    with pytest.raises(ValueError, match="absent from the calendar"):
        _base(challenger_ms, benchmark_ms, trimmed, modeling)


def test_a_tier_outside_the_configured_vocabulary_raises(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
) -> None:
    """Coercion nulls an unknown label, which then vanishes from every tier slice
    while still counting in global."""
    relabelled = challenger_ms.assign(tier=challenger_ms["tier"].astype(str))
    relabelled.loc[relabelled.index[0], "tier"] = "enormous"

    with pytest.raises(ValueError, match="not a prefix"):
        _base(relabelled, benchmark_ms, calendar_df, modeling)


def test_a_tier_set_that_skips_a_configured_label_raises(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
) -> None:
    """assign_tiers emits a prefix and never a gap, so a skipped label is drift."""
    top = _present_tier_labels(modeling)[-1]
    challenger = challenger_ms.assign(tier=top)
    benchmark = benchmark_ms.assign(tier=top)

    with pytest.raises(ValueError, match="not a prefix"):
        _base(challenger, benchmark, calendar_df, modeling)


def test_a_string_tier_is_coerced_to_the_configured_order(
    calendar_df: pd.DataFrame, modeling: TrainModelingConfig
) -> None:
    """A valid sidecar can carry one, and a string column sorts alphabetically."""
    roles = modeling.model_roles
    challenger = _monthly_series(roles.challenger, modeling, tier_as_string=True)
    benchmark = _monthly_series(roles.benchmark, modeling, tier_as_string=True)

    base = _base(challenger, benchmark, calendar_df, modeling)

    assert base["tier"].cat.ordered
    assert list(base["tier"].cat.categories) == list(_present_tier_labels(modeling))


def test_categories_are_compared_before_values_so_the_comparison_cannot_raise(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
) -> None:
    """Unreachable through the builder, which rebuilds both sides over one list.
    Removing that rebuild would turn the tier comparison into a bare TypeError."""
    base = _base(challenger_ms, benchmark_ms, calendar_df, modeling)
    restored = base.rename(
        columns={
            "tier": "tier_ch",
            "series_weight": "series_weight_ch",
            "actual_monthly_total": "actual_monthly_total_ch",
        }
    ).assign(
        tier_bm=base["tier"].cat.set_categories(["low", "high"]),
        series_weight_bm=base["series_weight"],
        actual_monthly_total_bm=base["actual_monthly_total"],
    )

    with pytest.raises(ValueError, match="tier categories differ"):
        _check_base_frame(restored)


# ================================================
# _score_folds: both models, every metric, at fold grain
# ================================================


@pytest.fixture(scope="module")
def base_frame(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
) -> pd.DataFrame:
    """The joined substrate, shared by the assertions below."""
    return _base(challenger_ms, benchmark_ms, calendar_df, modeling)


@pytest.fixture(scope="module")
def fold_metrics(
    base_frame: pd.DataFrame, modeling: TrainModelingConfig
) -> pd.DataFrame:
    """One scoring of the shaped fixture, shared by the assertions below."""
    return _score_folds(
        base_frame,
        challenger_model=modeling.model_roles.challenger,
        benchmark_model=modeling.model_roles.benchmark,
    )


def test_the_fold_table_has_one_row_per_model_tier_metric_and_fold(
    fold_metrics: pd.DataFrame, modeling: TrainModelingConfig
) -> None:
    """A duplicated cell would double an n_obs and skew every mean derived from
    it, while the table still looks square."""
    roles = modeling.model_roles

    assert not fold_metrics.duplicated(
        ["model", "horizon", "tier", "metric", *_FOLD_KEYS]
    ).any()
    assert set(fold_metrics["model"]) == {roles.challenger, roles.benchmark}


def test_both_models_carry_the_same_metric_set(
    fold_metrics: pd.DataFrame, modeling: TrainModelingConfig
) -> None:
    """Scoring the benchmark on the absolute metrics alone would make the set
    ragged by model, so any pivot on model has holes."""
    roles = modeling.model_roles
    challenger = set(
        fold_metrics.loc[fold_metrics["model"] == roles.challenger, "metric"]
    )
    benchmark = set(
        fold_metrics.loc[fold_metrics["model"] == roles.benchmark, "metric"]
    )

    assert challenger == benchmark
    assert len(challenger) > 1


def test_the_benchmark_scores_one_against_itself(
    fold_metrics: pd.DataFrame, modeling: TrainModelingConfig
) -> None:
    """The reference line on a skill chart, and a live check that the scoring path
    is wired. Exact for the pooled form, whose numerator and denominator are
    bit-identical sums, and to tolerance for the per-series form, which
    renormalizes weights and does return 0.9999999999999999 on this fixture."""
    benchmark = fold_metrics[fold_metrics["model"] == modeling.model_roles.benchmark]
    pooled = benchmark.loc[benchmark["metric"] == "wrmae_pooled", "value"].dropna()
    per_series = benchmark.loc[
        benchmark["metric"] == "wrmae_per_series", "value"
    ].dropna()

    assert not pooled.empty
    assert (pooled == 1.0).all()
    np.testing.assert_allclose(per_series, 1.0, rtol=1e-12)


def test_the_challenger_outscores_the_benchmark_in_every_cell(
    fold_metrics: pd.DataFrame, modeling: TrainModelingConfig
) -> None:
    """The fixture wins everywhere by construction, so a skill ratio that is not
    below 1.0 means the two sides were never actually compared: handing both
    models one forecast column reads as a clean 1.0 rather than as an error."""
    challenger = fold_metrics[fold_metrics["model"] == modeling.model_roles.challenger]
    relative = challenger.loc[
        challenger["metric"].isin(_RELATIVE_METRIC_FNS), "value"
    ].dropna()

    assert not relative.empty
    assert (relative < 1.0).all()


def test_the_relative_metrics_are_nan_in_the_same_cells_for_both_models(
    fold_metrics: pd.DataFrame, modeling: TrainModelingConfig
) -> None:
    """Both calls exclude on the benchmark's errors, so a cell where the benchmark
    reads 1.0 and the challenger reads nan is impossible under correct code."""
    roles = modeling.model_roles
    keys = ["horizon", "tier", "metric", *_FOLD_KEYS]
    relative = fold_metrics[fold_metrics["metric"].isin(_RELATIVE_METRIC_FNS)]

    challenger = relative[relative["model"] == roles.challenger].set_index(keys)[
        "value"
    ]
    benchmark = relative[relative["model"] == roles.benchmark].set_index(keys)["value"]
    benchmark = benchmark.reindex(challenger.index)

    assert challenger.isna().any()
    assert (challenger.isna() == benchmark.isna()).all()


def test_a_view_scores_the_same_as_the_sidecar_frame_it_came_from(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
) -> None:
    """The check that would have caught the row-set asymmetry. The relative
    metrics merge the two sides while the absolute ones never do, so a join that
    silently dropped rows would move one family and leave the other alone."""
    base = _base(challenger_ms, benchmark_ms, calendar_df, modeling)

    view = _model_view(base, "monthly_forecast_ch")

    np.testing.assert_allclose(compute_wape(view), compute_wape(challenger_ms))


def test_the_tier_slices_partition_the_global_row(
    fold_metrics: pd.DataFrame,
) -> None:
    """global is an aggregate beside the partition, not one of its members, which
    is the fact that makes summing across tier wrong."""
    keys = ["model", "metric", *_FOLD_KEYS]
    is_global = fold_metrics["tier"] == GLOBAL_TIER

    whole = fold_metrics[is_global].set_index(keys)["n_obs"].sort_index()
    parts = fold_metrics[~is_global].groupby(keys, observed=True)["n_obs"].sum()

    assert (whole == parts.reindex(whole.index)).all()


def test_a_tier_with_no_rows_in_a_fold_scores_nan(
    fold_metrics: pd.DataFrame,
) -> None:
    """The nan a nanmean drops one grain up. Without a fold missing a tier the
    derivation's exclusion behaviour is never exercised."""
    empty = fold_metrics[fold_metrics["n_obs"] == 0]

    assert not empty.empty
    assert empty["value"].isna().all()


def test_n_obs_counts_rows_available_rather_than_rows_used(
    fold_metrics: pd.DataFrame, base_frame: pd.DataFrame
) -> None:
    """Against an independent count, since rows used is the plausible wrong
    answer and it reads as a smaller number rather than as an error. One count
    per cell too: rows used would differ by metric, the fixture's dormant series
    being dropped by the per-series reductions and kept by wape."""
    available = base_frame.groupby(_FOLD_KEYS, observed=True).size().sort_index()
    counted = (
        fold_metrics[fold_metrics["tier"] == GLOBAL_TIER]
        .groupby(_FOLD_KEYS, observed=True)["n_obs"]
        .agg(["nunique", "max"])
        .sort_index()
    )

    assert (counted["nunique"] == 1).all()
    assert (counted["max"] == available).all()


def test_tier_is_ordered_with_the_global_row_first(
    fold_metrics: pd.DataFrame,
) -> None:
    """A dashboard reading one score table has no other ordering source, and the
    order below is the one an object column would not give."""
    categories = list(fold_metrics["tier"].cat.categories)

    assert fold_metrics["tier"].cat.ordered
    assert categories[0] == GLOBAL_TIER
    assert categories != sorted(categories)


def test_every_fold_carries_exactly_one_horizon(
    fold_metrics: pd.DataFrame,
) -> None:
    """Horizon is a function of the origin alone, which is why it does not
    multiply this table. A fold spanning two would double every derived cell."""
    per_fold = fold_metrics.groupby(_FOLD_KEYS, observed=True)["horizon"].nunique()

    assert (per_fold == 1).all()
    assert fold_metrics["horizon"].nunique() > 1


def test_one_model_in_both_roles_scores_once(
    challenger_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
) -> None:
    """The smoke test ModelRoles blesses. Scoring the name twice would double
    every n_obs while the table still looked well formed."""
    name = modeling.model_roles.challenger
    base = _base(
        challenger_ms, challenger_ms, calendar_df, modeling, benchmark_model=name
    )

    scored = _score_folds(base, challenger_model=name, benchmark_model=name)

    assert set(scored["model"]) == {name}
    assert not scored.duplicated(
        ["model", "horizon", "tier", "metric", *_FOLD_KEYS]
    ).any()
    pooled = scored.loc[scored["metric"] == "wrmae_pooled", "value"].dropna()
    assert (pooled == 1.0).all()


# ================================================
# The derivations: period and summary from the fold table
#
# The centerpiece: two independent paths, the metric itself against _derive.
# ================================================


@pytest.fixture(scope="module")
def outputs(
    challenger_ms: pd.DataFrame,
    benchmark_ms: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
) -> EvaluateOutputs:
    """One run of the pure function, shared by the assertions below."""
    return compute_evaluate_outputs(
        challenger_ms=challenger_ms,
        benchmark_ms=benchmark_ms,
        calendar_df=calendar_df,
        challenger_model=modeling.model_roles.challenger,
        benchmark_model=modeling.model_roles.benchmark,
        tier_labels=tuple(modeling.tiering.tier_labels),
        identity=_identity(),
    )


def _computed_directly(
    metric: str, challenger: pd.DataFrame, benchmark: pd.DataFrame, tier: str | None
) -> float:
    """The notebook's path: the metric gets the whole slice and averages its folds."""
    if metric == "wrmae_pooled":
        return compute_wrmae_pooled(challenger, benchmark, tier)
    return compute_wape(challenger, tier)


def _tier_filter(tier: object) -> str | None:
    """The metric functions take None for every series, not the sentinel."""
    return None if tier == GLOBAL_TIER else str(tier)


@pytest.mark.parametrize("metric", ["wrmae_pooled", "wape"])
def test_summary_equals_the_metric_computed_at_summary_grain(
    outputs: EvaluateOutputs,
    base_frame: pd.DataFrame,
    modeling: TrainModelingConfig,
    metric: str,
) -> None:
    """The identity a golden would only pin: one run over the whole horizon."""
    challenger = _model_view(base_frame, "monthly_forecast_ch")
    benchmark = _model_view(base_frame, "monthly_forecast_bm")
    summary = outputs.summary_metrics
    cells = summary[
        (summary["model"] == modeling.model_roles.challenger)
        & (summary["metric"] == metric)
    ]

    assert not cells.empty
    for cell in cells.itertuples():
        expected = _computed_directly(
            metric,
            challenger[challenger["horizon"] == cell.horizon],
            benchmark[benchmark["horizon"] == cell.horizon],
            _tier_filter(cell.tier),
        )
        if np.isnan(expected):
            assert np.isnan(cell.value)
        else:
            np.testing.assert_allclose(cell.value, expected, rtol=1e-12)


@pytest.mark.parametrize("metric", ["wrmae_pooled", "wape"])
def test_period_equals_the_metric_computed_within_its_month(
    outputs: EvaluateOutputs,
    base_frame: pd.DataFrame,
    modeling: TrainModelingConfig,
    metric: str,
) -> None:
    """Same construction one grain down: period derives by the same groupby."""
    challenger = _model_view(base_frame, "monthly_forecast_ch")
    benchmark = _model_view(base_frame, "monthly_forecast_bm")
    period = outputs.period_metrics
    cells = period[
        (period["model"] == modeling.model_roles.challenger)
        & (period["metric"] == metric)
    ]

    assert not cells.empty
    for cell in cells.itertuples():
        month = cell.predicted_fiscal_year_month
        in_cell = (challenger["horizon"] == cell.horizon) & (
            challenger["predicted_fiscal_year_month"] == month
        )
        expected = _computed_directly(
            metric, challenger[in_cell], benchmark[in_cell], _tier_filter(cell.tier)
        )
        if np.isnan(expected):
            assert np.isnan(cell.value)
        else:
            np.testing.assert_allclose(cell.value, expected, rtol=1e-12)


@pytest.mark.parametrize(
    ("table", "keys"),
    [("summary_metrics", _SCORE_KEYS), ("period_metrics", _PERIOD_KEYS)],
)
def test_the_derived_value_is_the_nanmean_of_its_fold_values(
    outputs: EvaluateOutputs, table: str, keys: list[str]
) -> None:
    """Every metric, and against numpy rather than the groupby the impl uses."""
    grouped = outputs.fold_metrics.groupby(keys, observed=True)["value"]
    with warnings.catch_warnings():
        # An all-nan group warns "Mean of empty slice" and returns nan, correctly.
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = {
            key: np.nanmean(values.to_numpy(dtype=float)) for key, values in grouped
        }

    derived = getattr(outputs, table)
    assert len(derived) == len(expected)
    for cell in derived.itertuples():
        key = tuple(getattr(cell, name) for name in keys)
        if np.isnan(expected[key]):
            assert np.isnan(cell.value)
        else:
            np.testing.assert_allclose(cell.value, expected[key], rtol=1e-12)


def test_n_obs_sums_across_folds_rather_than_averaging(
    outputs: EvaluateOutputs, base_frame: pd.DataFrame
) -> None:
    """A mean here yields a plausible frame with a wrong n_obs, and nothing else
    catches it."""
    available = base_frame.groupby("horizon", observed=True).size().sort_index()
    summary = outputs.summary_metrics
    counted = (
        summary[summary["tier"] == GLOBAL_TIER]
        .groupby("horizon", observed=True)["n_obs"]
        .agg(["nunique", "max"])
        .sort_index()
    )

    assert (counted["nunique"] == 1).all()
    assert (counted["max"] == available).all()
    # Otherwise a sum and a mean agree and the distinction is untested.
    assert outputs.fold_metrics.groupby(_SCORE_KEYS, observed=True).size().max() > 1


def test_n_folds_used_counts_only_the_folds_that_contributed(
    outputs: EvaluateOutputs,
) -> None:
    """Counted after exclusions where n_obs is counted before: not a pair."""
    fold = outputs.fold_metrics
    contributed = fold[fold["value"].notna()].groupby(_SCORE_KEYS, observed=True).size()
    covered = fold.groupby(_SCORE_KEYS, observed=True).size()
    reported = outputs.summary_metrics.set_index(_SCORE_KEYS)["n_folds_used"]

    assert (
        reported.sort_index() == contributed.reindex(reported.index).fillna(0)
    ).all()
    # Otherwise every fold contributed and the exclusion path is never exercised.
    assert (contributed.reindex(covered.index).fillna(0) < covered).any()


def test_a_cell_whose_every_fold_was_excluded_is_nan(
    outputs: EvaluateOutputs,
) -> None:
    """A nanmean over nothing is nan, and n_folds_used says so. Period grain only."""
    empty = outputs.period_metrics[outputs.period_metrics["n_folds_used"] == 0]

    assert not empty.empty
    assert empty["value"].isna().all()
    assert (empty["n_obs"] == 0).all()


def test_the_fixture_distinguishes_a_mean_of_folds_from_a_mean_of_periods(
    outputs: EvaluateOutputs,
) -> None:
    """A fixture where these agreed would pass the identity checks either way."""
    via_period = (
        outputs.period_metrics.groupby(_SCORE_KEYS, observed=True)["value"]
        .mean()
        .sort_index()
    )
    summary = outputs.summary_metrics.set_index(_SCORE_KEYS)["value"].sort_index()

    assert not np.isclose(summary, via_period, equal_nan=True).all()


@pytest.mark.parametrize(
    ("table", "keys"),
    [("summary_metrics", _SCORE_KEYS), ("period_metrics", _PERIOD_KEYS)],
)
def test_the_derived_cells_are_exactly_those_the_fold_table_carries(
    outputs: EvaluateOutputs, table: str, keys: list[str]
) -> None:
    """Unobserved category combinations would invent cells backed by no fold."""
    derived = getattr(outputs, table)
    from_fold = outputs.fold_metrics[keys].astype(str).drop_duplicates()

    assert set(map(tuple, derived[keys].astype(str).to_numpy())) == set(
        map(tuple, from_fold.to_numpy())
    )


def test_tier_survives_the_derivations_as_an_ordered_categorical(
    outputs: EvaluateOutputs,
) -> None:
    """A dropped dtype leaves a dashboard sorting high, low, middle, very_high."""
    for table in ("fold_metrics", "period_metrics", "summary_metrics"):
        tier = getattr(outputs, table)["tier"]
        categories = list(tier.cat.categories)

        assert tier.cat.ordered
        assert categories[0] == GLOBAL_TIER
        assert categories != sorted(categories)


def test_lineage_values_propagate_into_every_table(
    outputs: EvaluateOutputs,
) -> None:
    """The values, not that five columns exist."""
    identity = _identity()
    stamped = {
        "train_run_id": identity.train_run_id,
        "feature_run_id": identity.feature_run_id,
        "git_hash": identity.git_hash,
        "panel_uri": identity.panel_uri,
        "calendar_uri": identity.calendar_uri,
    }

    for table in (
        "per_series_comparison",
        "fold_metrics",
        "period_metrics",
        "summary_metrics",
    ):
        frame = getattr(outputs, table)
        for column, value in stamped.items():
            assert set(frame[column]) == {value}
