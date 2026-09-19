import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import yaml

from fcstnyctaxi.lib.config.bindings import (
    model_names_from_roles,
    train_modeling_bindings,
)
from fcstnyctaxi.lib.config.composition import compose_config, save_config
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

_SERIES = ("zone_a", "zone_b", "zone_c")

# Index into the present tier labels, per series and origin:
#
#   origin 0  2 bins:  a=low     b=very_low  c=very_low
#   origin 1  3 bins:  a=middle  b=low       c=very_low
#   origin 2  3 bins:  a=middle  b=very_low  c=low
#
# One fact drives it. zone_c is dormant at the first origin, so assign_tiers bins
# two series there and force-assigns it tier_labels[0], then bins three once it
# recovers. Nothing here is stipulated: the top tier has no series at origin 0,
# which is the empty slice a metric returns nan for; the category sets therefore
# differ by origin, which degrades a concatenated categorical to object; and two
# series migrate for two reasons, zone_a because the bin count moved and zone_b
# because zone_c's mean of positive weeks overtook it.
#
# Three and not two: qcut spreads across every bin it makes, so
# min(len(tier_labels), mean_pos.nunique()) is 3 whenever all three are active.
_TIER_INDEX = {
    "zone_a": (1, 2, 2),
    "zone_b": (0, 1, 0),
    "zone_c": (0, 0, 1),
}

# cbrt of a 26-week trailing sum, clipped at zero, so it moves with the origin.
# zone_c carries none while dormant, the cell the per-series reductions used to
# score as a perfect forecast. It stays under zone_b's even after its tier passes
# it: tier averages the positive weeks alone, weight the whole window.
_WEIGHT = {
    "zone_a": (29.6, 29.6, 29.6),
    "zone_b": (17.3, 17.3, 17.3),
    "zone_c": (0.0, 6.7, 10.3),
}

# What each series realizes: a fact about the month, not the origin. zone_c's is
# positive throughout, since tier and weight look back while the actual looks
# forward, so a dormant series still has revenue to forecast.
_ACTUAL_LEVEL = {"zone_a": 1000.0, "zone_b": 200.0, "zone_c": 50.0}


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
    units would leave the normalization unexercised, and a mismatch on the join
    key makes every joined row one-sided.
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
    """The normalization needs work to do: a mismatch one-sides every row."""
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
