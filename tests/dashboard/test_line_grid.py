"""The line grids: the weekly reindex, the month rule, `global`, and one facet
drawn at either grain.

The weekly panel is hand-built, since the Feature pipeline owns the real one; its
self-check is that every row falls on a calendar week. Each model's weekly
forecasts are hand-built too, and its `monthly_series` is assembled from them and
the panel as backtest assembles each fold, tiers and weights included, so the two
grains agree. `per_series_comparison` comes from `compute_evaluate_outputs`.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots
from streamlit.testing.v1 import AppTest

from dashboard.shared import axes, palette
from dashboard.train import line_grid
from fcstnyctaxi.core.train import backtest_impl
from fcstnyctaxi.core.train.evaluate_impl import compute_evaluate_outputs
from fcstnyctaxi.lib.monthly_aggregation import (
    attach_tier_and_weight,
    build_monthly_forecast_vs_actual,
    compute_actual_monthly_totals,
)
from fcstnyctaxi.lib.period_utils import assign_tiers, compute_series_weights
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity

CHALLENGER = "model_a"
BENCHMARK = "model_b"
TIER_LABELS = ("low", "high")

# Half a 4/4/5 fiscal year of Sunday weeks.
WEEKS_IN_MONTH = {202401: 4, 202402: 4, 202403: 5, 202404: 4, 202405: 4, 202406: 5}
WEEKS = pd.date_range("2023-12-31", periods=sum(WEEKS_IN_MONTH.values()), freq="W-SUN")
FULL = range(len(WEEKS))
GAP_WEEK = 9
"""The second week of 202403."""
MID_MONTH_START = 6
"""The third week of 202402."""

# The series without a fold at the first origin starts after it, in 202404's
# second week; the second origin, in 202404's third, has folds for all three.
LATE = "time_series_c"
SPANS = {"time_series_a": FULL, "time_series_b": FULL, LATE: range(14, len(WEEKS))}
FIRST_ORIGIN = WEEKS[GAP_WEEK]
SECOND_ORIGIN = WEEKS[15]
ORIGIN_TARGETS = (
    (FIRST_ORIGIN, (202403, 202404)),
    (SECOND_ORIGIN, (202404, 202405)),
)
HISTORY_GAP = 2
"""The third week of 202401, before the first origin's target months."""
WINDOW_GAP = 13
"""The first week of 202404, a month the first origin forecasts."""
ROLE_MODELS = {"challenger": CHALLENGER, "benchmark": BENCHMARK}


def _calendar() -> pd.DataFrame:
    """The calendar columns the reindex, the month rule and evaluate read."""
    rows = [
        {
            "fiscal_year_month": month,
            "fiscal_week_of_month": week,
            "weeks_in_month": weeks,
            "origin_month_fraction_elapsed": week / weeks,
        }
        for month, weeks in WEEKS_IN_MONTH.items()
        for week in range(1, weeks + 1)
    ]
    return pd.DataFrame(rows).assign(ds=WEEKS)


def _snapshot(
    spans: dict[str, range],
    *,
    nan: frozenset = frozenset(),
    absent: frozenset = frozenset(),
) -> pd.DataFrame:
    """A weekly panel as Feature ships it. `nan` keeps a (series, week) row with a
    null `y`; `absent` drops the row."""
    rows = [
        {
            "unique_id": series,
            "ds": WEEKS[week],
            "y": np.nan if (series, week) in nan else 100.0 * (index + 1) + week,
        }
        for index, (series, weeks) in enumerate(spans.items())
        for week in weeks
        if (series, week) not in absent
    ]
    frame = pd.DataFrame(rows)
    # Self-check: every row is a calendar week, as in Feature's panel.
    assert set(frame["ds"]) <= set(WEEKS)
    return frame


def _months(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Each series' months under the month rule."""
    calendar = _calendar()
    return line_grid.monthly_totals(
        line_grid.reindex_weeks(snapshot, calendar), calendar
    )


def _by_month(months: pd.DataFrame, series: str) -> pd.DataFrame:
    """One series' months, indexed by month."""
    return months[months["unique_id"] == series].set_index("fiscal_year_month")


def _raw_cv_forecasts(model: str, snapshot: pd.DataFrame) -> pd.DataFrame:
    """One sidecar's weekly forecasts: from each origin, for every series with a
    training row, each week to its later target month's end."""
    last_week = _calendar().groupby("fiscal_year_month")["ds"].max()
    error = 0.20 if model == BENCHMARK else 0.10
    rows = []
    for fold, (origin, months) in enumerate(ORIGIN_TARGETS):
        weeks = WEEKS[(WEEKS > origin) & (WEEKS <= last_week[months[-1]])]
        train = snapshot[snapshot["ds"] <= origin]
        for series, level in train.groupby("unique_id")["y"].last().items():
            rows += [
                {
                    "forecast_origin_date": origin,
                    "unique_id": series,
                    "ds": week,
                    "ypred": level * (1 + error + 0.01 * step),
                    "fold_id": f"fold_{fold}",
                }
                for step, week in enumerate(weeks)
            ]
    return pd.DataFrame(rows)[backtest_impl._RAW_CV_FORECAST_COLUMNS]


def _monthly_series(snapshot: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """One sidecar's monthly_series, assembled from its weekly forecasts and the
    panel as backtest assembles each fold."""
    calendar = _calendar()
    actuals = compute_actual_monthly_totals(snapshot, calendar)
    fraction = calendar.set_index("ds")["origin_month_fraction_elapsed"]
    folds = []
    for origin, _ in ORIGIN_TARGETS:
        train = snapshot[snapshot["ds"] <= origin]
        folds.append(
            attach_tier_and_weight(
                monthly_rows_df=build_monthly_forecast_vs_actual(
                    forecast_df=raw[raw["forecast_origin_date"] == origin],
                    train_df=train,
                    calendar_df=calendar,
                    actual_monthly_df=actuals,
                ),
                tier_df=assign_tiers(train, origin, calendar, tier_labels=TIER_LABELS),
                weight_df=compute_series_weights(train, origin, calendar),
                fold_origin=origin,
                origin_month_fraction_elapsed=fraction[origin],
            )
        )
    return pd.concat(folds, ignore_index=True)


def _grid(
    snapshot: pd.DataFrame, models: dict[str, str] = ROLE_MODELS
) -> line_grid.GridData:
    """Every frame a facet reads, each through its producer, over `snapshot`."""
    calendar = _calendar()
    raw = {role: _raw_cv_forecasts(model, snapshot) for role, model in models.items()}
    identity = TrainRunIdentity(
        git_hash="0" * 40,
        feature_run_id="feature_run",
        train_run_id="train-20250101t000000000000z",
        panel_uri="gs://bucket/panel.parquet",
        calendar_uri="gs://bucket/calendar.parquet",
        additional_exog_uri="gs://bucket/exog.parquet",
    )
    table = compute_evaluate_outputs(
        challenger_ms=_monthly_series(snapshot, raw["challenger"]),
        benchmark_ms=_monthly_series(snapshot, raw["benchmark"]),
        calendar_df=calendar,
        challenger_model=models["challenger"],
        benchmark_model=models["benchmark"],
        tier_labels=TIER_LABELS,
        identity=identity,
    ).per_series_comparison
    weekly = line_grid.reindex_weeks(snapshot, calendar)
    return line_grid.GridData(
        calendar=calendar,
        weekly=weekly,
        monthly=line_grid.monthly_totals(weekly, calendar),
        per_series_comparison=table,
        raw_cv_forecasts=raw,
        models=models,
    )


@pytest.fixture(scope="module")
def grid() -> line_grid.GridData:
    """The frames over the panel with a late series."""
    data = _grid(_snapshot(SPANS))
    table = data.per_series_comparison
    # Self-check: the late series has folds, none at the first origin, and one
    # predicting a month the first origin also predicts.
    late = table[table["unique_id"] == LATE]
    assert not late.empty
    assert FIRST_ORIGIN not in set(late["forecast_origin_date"])
    first = table[table["forecast_origin_date"] == FIRST_ORIGIN]
    assert set(late["predicted_fiscal_year_month"]) & set(
        first["predicted_fiscal_year_month"]
    )
    # Self-check: a monthly forecast is its month's weeks to the origin, actual,
    # plus its forecast weeks, as backtest assembles it.
    raw = data.raw_cv_forecasts["challenger"]
    week_month = _calendar().set_index("ds")["fiscal_year_month"]
    row = first[first["predicted_fiscal_year_month"] == 202403].iloc[0]
    weekly = data.weekly[data.weekly["unique_id"] == row["unique_id"]]
    observed = weekly[
        (weekly["fiscal_year_month"] == 202403) & (weekly["ds"] <= FIRST_ORIGIN)
    ]["y"].sum()
    forecast = raw[
        (raw["forecast_origin_date"] == FIRST_ORIGIN)
        & (raw["unique_id"] == row["unique_id"])
    ]
    predicted = forecast[forecast["ds"].map(week_month) == 202403]["ypred"].sum()
    assert row["monthly_forecast_ch"] == pytest.approx(observed + predicted)
    return data


@pytest.fixture(scope="module")
def per_series(grid: line_grid.GridData) -> pd.DataFrame:
    """The producer's per_series_comparison over the panel with a late series."""
    return grid.per_series_comparison


# ================================================
# the month rule
# ================================================


def test_a_nan_week_and_an_absent_row_gap_their_month_alike() -> None:
    """Each leaves a week inside the span without a value, so neither reads whole."""
    months = _months(
        _snapshot(
            {"time_series_a": FULL, "time_series_b": FULL},
            nan=frozenset({("time_series_a", GAP_WEEK)}),
            absent=frozenset({("time_series_b", GAP_WEEK)}),
        )
    )
    nan_week = _by_month(months, "time_series_a")
    absent_row = _by_month(months, "time_series_b")
    assert nan_week["status"].equals(absent_row["status"])
    assert nan_week.loc[202403, "status"] == line_grid.GAPPED
    assert np.isnan(nan_week.loc[202403, "y"])


def test_a_series_starting_mid_month_drops_its_first_month_without_a_gap() -> None:
    """Nothing is missing there, the series had not begun, so no note is owed."""
    # Self-check: the series' first week is not its month's first.
    assert _calendar()["fiscal_week_of_month"][MID_MONTH_START] != 1
    months = _by_month(
        _months(_snapshot({"time_series_a": range(MID_MONTH_START, len(WEEKS))})),
        "time_series_a",
    )
    assert months.loc[202402, "status"] == line_grid.INCOMPLETE
    assert np.isnan(months.loc[202402, "y"])
    assert months.loc[202403, "status"] == line_grid.COMPLETE


def test_a_panel_ending_mid_month_drops_its_last_month() -> None:
    """The weeks after the panel's end are not yet known, so the sum is partial."""
    end = len(WEEKS) - 2
    # Self-check: the panel ends before its last month does.
    assert _calendar()["origin_month_fraction_elapsed"][end - 1] < 1
    months = _months(
        _snapshot({"time_series_a": range(end), "time_series_b": range(end)})
    )
    for series in ("time_series_a", "time_series_b"):
        assert _by_month(months, series).loc[202406, "status"] == line_grid.INCOMPLETE
        assert _by_month(months, series).loc[202405, "status"] == line_grid.COMPLETE


def test_monthly_x_is_the_months_last_fiscal_week(per_series) -> None:
    """Read from the calendar, so the gap from the origin rule to it is the MTD."""
    calendar = _calendar()
    last_week = calendar.groupby("fiscal_year_month")["ds"].max()
    months = _months(_snapshot(SPANS))
    points = line_grid.forecast_points(
        per_series, FIRST_ORIGIN, line_grid.fold_set(per_series, FIRST_ORIGIN), calendar
    )
    assert months["ds"].equals(months["fiscal_year_month"].map(last_week))
    assert points["ds"].equals(points["predicted_fiscal_year_month"].map(last_week))


# ================================================
# global
# ================================================


def test_global_is_gapped_when_one_member_is() -> None:
    """Its true total is unknown, whatever the other members hold."""
    members = ("time_series_a", "time_series_b")
    months = _months(
        _snapshot(dict.fromkeys(members, FULL), nan=frozenset({(members[0], GAP_WEEK)}))
    )
    # Self-check: the other member's month is whole, so only the gap marks it.
    assert _by_month(months, members[1]).loc[202403, "status"] == line_grid.COMPLETE
    totals = line_grid.global_monthly(months, members).set_index("fiscal_year_month")
    assert totals.loc[202403, "status"] == line_grid.GAPPED
    assert np.isnan(totals.loc[202403, "y"])


def test_global_leaves_out_a_members_incomplete_month() -> None:
    """As that member's own facet does: global sums the members whose month is whole."""
    months = _months(
        _snapshot(
            {
                "time_series_a": FULL,
                "time_series_b": range(MID_MONTH_START, len(WEEKS)),
            }
        )
    )
    totals = line_grid.global_monthly(
        months, ("time_series_a", "time_series_b")
    ).set_index("fiscal_year_month")
    whole = _by_month(months, "time_series_a")
    assert totals.loc[202402, "status"] == line_grid.COMPLETE
    assert totals.loc[202402, "y"] == whole.loc[202402, "y"]


def test_a_series_without_a_fold_at_the_origin_is_left_out_of_global(
    per_series,
) -> None:
    """From both sides: global's actual and forecast cover the same series."""
    members = line_grid.fold_set(per_series, FIRST_ORIGIN)
    assert LATE not in members

    months = _months(_snapshot(SPANS))
    # Self-check: the late series has whole months global would otherwise sum.
    assert (_by_month(months, LATE)["status"] == line_grid.COMPLETE).any()
    member_months = months[months["unique_id"].isin(members)]
    whole = member_months[member_months["status"] == line_grid.COMPLETE]
    expected = whole.groupby("fiscal_year_month")["y"].sum()
    totals = line_grid.global_monthly(months, members).set_index("fiscal_year_month")
    assert totals.loc[expected.index, "y"].tolist() == pytest.approx(expected.tolist())

    points = line_grid.forecast_points(per_series, FIRST_ORIGIN, members, _calendar())
    at_origin = per_series[per_series["forecast_origin_date"] == FIRST_ORIGIN]
    expected = at_origin.groupby("predicted_fiscal_year_month")["monthly_forecast_ch"]
    assert points.set_index("predicted_fiscal_year_month")[
        "monthly_forecast_ch"
    ].tolist() == pytest.approx(expected.sum().tolist())


# ================================================
# one facet
# ================================================

HORIZONS = ("horizon_1", "horizon_2")
GRAINS = (line_grid.MONTHLY, line_grid.WEEKLY)


def _month_of(week: int) -> int:
    """The fiscal month a week index falls in."""
    return int(_calendar()["fiscal_year_month"].iloc[week])


@pytest.fixture(scope="module")
def gapped_grid() -> line_grid.GridData:
    """The frames over the panel with time_series_a missing two weeks: a NaN one
    before the first origin's target months and an absent one inside them."""
    data = _grid(
        _snapshot(
            SPANS,
            nan=frozenset({("time_series_a", HISTORY_GAP)}),
            absent=frozenset({("time_series_a", WINDOW_GAP)}),
        )
    )
    # Self-check: only the second gap falls in a month the first origin forecasts.
    table = data.per_series_comparison
    targets = set(
        table.loc[
            table["forecast_origin_date"] == FIRST_ORIGIN, "predicted_fiscal_year_month"
        ]
    )
    assert _month_of(WINDOW_GAP) in targets
    assert _month_of(HISTORY_GAP) not in targets
    return data


def _drawn(
    data: line_grid.GridData, series: str, origin: pd.Timestamp, grain: str
) -> go.Figure:
    """One facet, alone in a figure."""
    figure = make_subplots(rows=1, cols=1)
    facet = line_grid.build_facet(data, series, origin, grain)
    line_grid.add_facet(figure, facet, data.models, row=1, col=1)
    return figure


def _forecasts(figure: go.Figure) -> dict[tuple[str, str], go.Scatter]:
    """Each role's forecast trace per horizon, found by its legend name."""
    return {
        (role, horizon): trace
        for role, model in ROLE_MODELS.items()
        for horizon in HORIZONS
        for trace in figure.select_traces(selector={"name": f"{model} {horizon}"})
    }


def _named(figure: go.Figure, name: str) -> list:
    """The figure's annotations carrying `name`."""
    return [
        annotation
        for annotation in figure.layout.annotations
        if annotation.name == name
    ]


def _drawn_actual(figure: go.Figure) -> pd.DataFrame:
    """The actual trace's points, with each one's fiscal month."""
    (actual,) = figure.select_traces(selector={"name": line_grid.ACTUAL})
    week_month = _calendar().set_index("ds")["fiscal_year_month"]
    ds = pd.to_datetime(pd.Series(actual.x))
    return pd.DataFrame(
        {"ds": ds, "month": ds.map(week_month), "y": np.asarray(actual.y, dtype=float)}
    )


def _shape(trace: go.Scatter) -> str:
    """A marker's symbol without its fill."""
    return trace.marker.symbol.removesuffix("-open")


def _is_open(trace: go.Scatter) -> bool:
    """Whether a marker is drawn unfilled."""
    return trace.marker.symbol.endswith("-open")


@pytest.mark.parametrize("grain", GRAINS)
def test_color_and_shape_carry_horizon_while_dash_and_fill_carry_the_model(
    grid, grain: str
) -> None:
    """So each of the two dimensions has a channel besides color."""
    traces = _forecasts(_drawn(grid, "time_series_a", FIRST_ORIGIN, grain))
    # Self-check: both roles are drawn at both horizons.
    assert set(traces) == {(role, h) for role in ROLE_MODELS for h in HORIZONS}
    for horizon in HORIZONS:
        challenger, benchmark = (
            traces["challenger", horizon],
            traces["benchmark", horizon],
        )
        colors = {challenger.marker.color, benchmark.marker.color}
        assert colors == {challenger.line.color, benchmark.line.color}
        assert len(colors) == 1
        assert _shape(challenger) == _shape(benchmark)
        assert challenger.line.dash != benchmark.line.dash
        assert _is_open(benchmark) and not _is_open(challenger)
    for role in ROLE_MODELS:
        first, second = (traces[role, horizon] for horizon in HORIZONS)
        assert first.marker.color != second.marker.color
        assert _shape(first) != _shape(second)
        assert first.line.dash == second.line.dash
        assert _is_open(first) == _is_open(second)


def test_each_benchmark_trace_is_drawn_after_its_challenger(grid) -> None:
    """So the open ring lands on top of the filled dot it may coincide with."""
    figure = _drawn(grid, "time_series_a", FIRST_ORIGIN, line_grid.WEEKLY)
    position = {id(trace): index for index, trace in enumerate(figure.data)}
    traces = _forecasts(figure)
    for horizon in HORIZONS:
        assert (
            position[id(traces["benchmark", horizon])]
            > position[id(traces["challenger", horizon])]
        )


def test_a_coincidence_reads_as_a_ring_around_a_dot(grid) -> None:
    """The ring clears the dot's edge; at equal sizes the ring sits on that edge."""
    traces = _forecasts(_drawn(grid, "time_series_a", FIRST_ORIGIN, line_grid.MONTHLY))
    for horizon in HORIZONS:
        dot = traces["challenger", horizon].marker
        ring = traces["benchmark", horizon].marker
        # A stroke straddles the marker's edge.
        assert (ring.size - ring.line.width) / 2 > (dot.size + dot.line.width) / 2


def _origin_rule(figure: go.Figure):
    """The facet's one dash-dot rule."""
    (rule,) = [shape for shape in figure.layout.shapes if shape.line.dash == "dashdot"]
    return rule


@pytest.mark.parametrize("grain", GRAINS)
def test_the_origin_rule_sits_at_the_origin_date(grid, grain: str) -> None:
    """At full height, so its gap to horizon_1's month end is the month to date."""
    rule = _origin_rule(_drawn(grid, "time_series_a", FIRST_ORIGIN, grain))
    assert pd.Timestamp(rule.x0) == pd.Timestamp(rule.x1) == FIRST_ORIGIN
    assert (rule.y0, rule.y1, rule.yref) == (0, 1, "y domain")


def test_the_rules_draw_opaque(grid) -> None:
    """The page's template draws every shape at 0.3, which all but hides a rule."""
    figure = _drawn(grid, "time_series_a", FIRST_ORIGIN, line_grid.WEEKLY)
    figure.update_layout(**axes.base_layout())
    shapes = figure.full_figure_for_development(warn=False).layout.shapes
    # Self-check: the month boundary's tick is drawn beside the origin rule.
    assert len(shapes) > 1
    assert all(shape.opacity == 1 for shape in shapes)


def test_a_weekly_run_changes_color_midway_into_its_next_month(
    grid, per_series
) -> None:
    """Each week wears its own month's horizon, and the line turns where the tick is."""
    at_origin = per_series[per_series["forecast_origin_date"] == FIRST_ORIGIN]
    horizon_of = (
        at_origin.drop_duplicates("predicted_fiscal_year_month")
        .set_index("predicted_fiscal_year_month")["horizon"]
        .astype(str)
    )
    calendar = _calendar()
    week_month = calendar.set_index("ds")["fiscal_year_month"]
    earlier, later = dict(ORIGIN_TARGETS)[FIRST_ORIGIN]
    last = calendar.loc[calendar["fiscal_year_month"] == earlier, "ds"].max()
    first = calendar.loc[calendar["fiscal_year_month"] == later, "ds"].min()
    boundary = last + (first - last) / 2

    figure = _drawn(grid, "time_series_a", FIRST_ORIGIN, line_grid.WEEKLY)
    for (_, horizon), trace in _forecasts(figure).items():
        months = pd.to_datetime(pd.Series(trace.x)).map(week_month)
        assert {horizon_of[month] for month in months} == {horizon}
    bridges = [trace for trace in figure.data if trace.hoverinfo == "skip"]
    # Self-check: the run crosses the boundary, so there are halves on both sides.
    assert {horizon_of[week_month[pd.Timestamp(b.x[0])]] for b in bridges} == set(
        HORIZONS
    )
    for bridge in bridges:
        side = horizon_of[week_month[pd.Timestamp(bridge.x[0])]]
        assert pd.Timestamp(bridge.x[-1]) == boundary
        assert bridge.line.color == palette.rgba(palette.HORIZON_COLOR[side])
    # Each role's halves meet midway along the straight segment between the weeks.
    traces = _forecasts(figure)
    for role in ROLE_MODELS:
        before, after = (traces[role, horizon] for horizon in HORIZONS)
        halves = [b for b in bridges if b.line.dash == before.line.dash]
        middle = (before.y[-1] + after.y[0]) / 2
        assert [b.y[-1] for b in halves] == pytest.approx([middle] * len(halves))
    (tick,) = [shape for shape in figure.layout.shapes if shape.line.dash != "dashdot"]
    assert pd.Timestamp(tick.x0) == boundary


def test_a_monthly_forecast_is_its_points_alone(grid) -> None:
    """An origin predicts two months and nothing between them, so no line joins them."""
    figure = _drawn(grid, "time_series_a", FIRST_ORIGIN, line_grid.MONTHLY)
    # Self-check: both horizons are drawn, so a bridge would have two months to join.
    assert {horizon for _, horizon in _forecasts(figure)} == set(HORIZONS)
    assert not [trace for trace in figure.data if trace.hoverinfo == "skip"]
    assert len(figure.layout.shapes) == 1


@pytest.mark.parametrize("grain", GRAINS)
def test_an_empty_intersection_draws_the_history_and_says_why(grid, grain: str) -> None:
    """A series with no fold at the origin, which an empty frame shows as broken."""
    # Self-check: the late series has history but no fold at the first origin.
    assert LATE not in line_grid.fold_set(grid.per_series_comparison, FIRST_ORIGIN)
    facet = line_grid.build_facet(grid, LATE, FIRST_ORIGIN, grain)
    assert facet.forecast.empty
    assert line_grid.NO_FORECAST in facet.notes

    figure = _drawn(grid, LATE, FIRST_ORIGIN, grain)
    assert _drawn_actual(figure)["y"].notna().any()
    assert not _forecasts(figure)
    (note,) = _named(figure, line_grid.NOTE_ANNOTATION)
    assert line_grid.NO_FORECAST in note.text


@pytest.mark.parametrize("series", ["time_series_a", line_grid.GLOBAL])
@pytest.mark.parametrize("grain", GRAINS)
def test_a_gapped_month_draws_no_point_and_the_note_names_it(
    gapped_grid, grain: str, series: str
) -> None:
    """Before the target months and inside them, where the note also says why."""
    figure = _drawn(gapped_grid, series, FIRST_ORIGIN, grain)
    drawn = _drawn_actual(figure)
    for week in (HISTORY_GAP, WINDOW_GAP):
        if grain == line_grid.MONTHLY:
            at = drawn[drawn["month"] == _month_of(week)]
        else:
            at = drawn[drawn["ds"] == WEEKS[week]]
        # Present and missing, so the line breaks there rather than bridging it.
        assert len(at) == 1
        assert at["y"].isna().all()
    (note,) = _named(figure, line_grid.NOTE_ANNOTATION)
    assert str(_month_of(HISTORY_GAP)) in note.text
    assert f"{_month_of(WINDOW_GAP)} scored against a short total" in note.text
    assert ("a member" in note.text) == (series == line_grid.GLOBAL)
    if grain == line_grid.WEEKLY:
        # The month's other weeks are drawn, so the note must not say otherwise.
        month = drawn[drawn["month"] == _month_of(WINDOW_GAP)]
        assert month["y"].notna().any()
        assert "not drawn" not in note.text


def test_the_notes_sit_above_the_plot_under_the_title(gapped_grid) -> None:
    """On a header line of their own, so no note covers data."""
    figure = _drawn(gapped_grid, "time_series_a", FIRST_ORIGIN, line_grid.MONTHLY)
    (note,) = _named(figure, line_grid.NOTE_ANNOTATION)
    (title,) = _named(figure, line_grid.TITLE_ANNOTATION)
    assert note.yref.endswith("domain")
    assert note.y >= 1
    assert note.yanchor == "bottom"
    assert (title.yshift or 0) > (note.yshift or 0)


def test_globals_title_prints_its_fold_set_size(grid) -> None:
    """Membership moves with the origin, so the title says how many series it sums."""
    sizes = {
        origin: len(line_grid.fold_set(grid.per_series_comparison, origin))
        for origin, _ in ORIGIN_TARGETS
    }
    # Self-check: the two origins' fold sets differ in size.
    assert len(set(sizes.values())) == len(sizes)
    for origin, size in sizes.items():
        facet = line_grid.build_facet(grid, line_grid.GLOBAL, origin, line_grid.MONTHLY)
        assert f"({size} series" in facet.title


@pytest.mark.parametrize("series", ["time_series_a", line_grid.GLOBAL])
def test_a_weekly_forecast_sums_the_facets_members(grid, series: str) -> None:
    """One series' own weeks, or each week summed over global's fold set."""
    members = (
        line_grid.fold_set(grid.per_series_comparison, FIRST_ORIGIN)
        if series == line_grid.GLOBAL
        else (series,)
    )
    # Self-check: global sums more than one series here.
    assert len(line_grid.fold_set(grid.per_series_comparison, FIRST_ORIGIN)) > 1
    facet = line_grid.build_facet(grid, series, FIRST_ORIGIN, line_grid.WEEKLY)
    for role, raw in grid.raw_cv_forecasts.items():
        weeks = raw[raw["forecast_origin_date"] == FIRST_ORIGIN].pivot(
            index="ds", columns="unique_id", values="ypred"
        )
        expected = weeks[list(members)].sum(axis=1)
        assert facet.forecast[role].tolist() == pytest.approx(expected.tolist())


def test_a_late_member_adds_to_global_weekly_only_from_its_first_week(grid) -> None:
    """Before it, global's weeks are the other members' sum, with nothing missing."""
    # Self-check: the late series is a member at the second origin.
    assert LATE in line_grid.fold_set(grid.per_series_comparison, SECOND_ORIGIN)
    facet = line_grid.build_facet(
        grid, line_grid.GLOBAL, SECOND_ORIGIN, line_grid.WEEKLY
    )
    total = facet.actual.set_index("ds")["y"]
    weeks = grid.weekly.pivot(index="ds", columns="unique_id", values="y")
    before, start = WEEKS[SPANS[LATE].start - 1], WEEKS[SPANS[LATE].start]
    assert total[before] == pytest.approx(weeks.loc[before].drop(LATE).sum())
    assert total[start] == pytest.approx(weeks.loc[start].sum())


def test_a_missing_week_in_one_member_leaves_globals_week_missing(gapped_grid) -> None:
    """Its true total is unknown, whatever the other members hold."""
    facet = line_grid.build_facet(
        gapped_grid, line_grid.GLOBAL, FIRST_ORIGIN, line_grid.WEEKLY
    )
    total = facet.actual.set_index("ds")["y"]
    assert np.isnan(total[WEEKS[HISTORY_GAP]])
    assert np.isfinite(total[WEEKS[HISTORY_GAP + 1]])


# ================================================
# two grids
# ================================================

SIX = [
    ("time_series_a", FIRST_ORIGIN),
    ("time_series_b", FIRST_ORIGIN),
    (line_grid.GLOBAL, FIRST_ORIGIN),
    ("time_series_a", SECOND_ORIGIN),
    (LATE, SECOND_ORIGIN),
    (line_grid.GLOBAL, SECOND_ORIGIN),
]


def _grid_figure(data, selections=SIX, *, shared_y=False, window=False) -> go.Figure:
    """A monthly grid of six facets."""
    return line_grid.grid_figure(
        data, selections, line_grid.MONTHLY, shared_y=shared_y, window=window
    )


def _axes(figure: go.Figure, prefix: str) -> list:
    """The figure's x or y axes in facet order, which the layout's own is not."""
    names = [name for name in figure.layout if name.startswith(prefix)]
    return [figure.layout[name] for name in sorted(names, key=_facet_number)]


def _facet_number(axis_name: str) -> int:
    """`yaxis` is facet 1, `yaxis2` facet 2."""
    return int(axis_name.removeprefix("xaxis").removeprefix("yaxis") or 1)


def _plotted(figure: go.Figure, axis: str) -> np.ndarray:
    """Every finite value drawn against one y axis."""
    values = [
        y
        for trace in figure.data
        if (trace.yaxis or "y") == axis
        for y in np.asarray(trace.y, dtype=float)
    ]
    return np.array([value for value in values if np.isfinite(value)])


def test_a_grids_facets_share_one_x_range_covering_full_history(grid) -> None:
    """One zoom moves all six, and a short series draws only over the right."""
    figure = _grid_figure(grid)
    xaxes = _axes(figure, "xaxis")
    anchors = [axis for axis in xaxes if axis.matches is None]
    assert len(anchors) == 1
    assert {axis.matches for axis in xaxes} - {None} == {
        anchors[0].anchor.replace("y", "x")
    }
    ((low, high),) = {tuple(axis.range) for axis in xaxes}
    assert pd.Timestamp(low) <= grid.weekly["ds"].min()
    assert pd.Timestamp(high) >= grid.weekly["ds"].max()


def test_the_backtest_window_keeps_every_forecast_in_view(grid) -> None:
    """From before the run's first origin to the panel's end."""
    figure = _grid_figure(grid, window=True)
    ((low, high),) = {tuple(axis.range) for axis in _axes(figure, "xaxis")}
    first = grid.per_series_comparison["forecast_origin_date"].min()
    assert pd.Timestamp(low) < first
    assert pd.Timestamp(high) >= grid.weekly["ds"].max()


@pytest.fixture
def spiked_grid(monkeypatch: pytest.MonkeyPatch) -> line_grid.GridData:
    """A spike early in time_series_a's history, with the window shortened so it
    starts inside the fixture's half year, after the spike."""
    snapshot = _snapshot(SPANS)
    spike = (snapshot["unique_id"] == "time_series_a") & (snapshot["ds"] == WEEKS[0])
    snapshot.loc[spike, "y"] = 1e6
    monkeypatch.setattr(line_grid, "_HISTORY", pd.Timedelta(weeks=4))
    # Self-check: the shortened window starts after the spike.
    assert FIRST_ORIGIN - line_grid._HISTORY > WEEKS[0]
    return _grid(snapshot)


@pytest.mark.parametrize("shared_y", [False, True])
def test_the_backtest_window_fits_y_to_what_it_shows(spiked_grid, shared_y) -> None:
    """Plotly fits y to all of a facet's data, so a spike out of view flattens it."""
    figure = _grid_figure(spiked_grid, shared_y=shared_y, window=True)
    ((low, _),) = {tuple(axis.range) for axis in _axes(figure, "xaxis")}
    assert pd.Timestamp(low) > WEEKS[0]
    (facet_a,) = _axes(figure, "yaxis")[:1]
    assert facet_a.range is not None
    assert facet_a.range[1] < 1e6
    in_view = [
        y
        for trace in figure.data
        if (trace.yaxis or "y") == "y"
        for x, y in zip(pd.to_datetime(pd.Series(trace.x)), trace.y, strict=True)
        if x >= pd.Timestamp(low) and np.isfinite(y)
    ]
    assert facet_a.range[0] <= min(in_view)
    assert max(in_view) <= facet_a.range[1]


def test_shared_y_sets_one_range_holding_every_value(grid) -> None:
    """So the six read on one scale."""
    figure = _grid_figure(grid, shared_y=True)
    ((low, high),) = {tuple(axis.range) for axis in _axes(figure, "yaxis")}
    for index, _ in enumerate(SIX):
        values = _plotted(figure, "y" if index == 0 else f"y{index + 1}")
        assert low <= values.min()
        assert values.max() <= high


def test_independent_y_at_full_history_sets_none(grid) -> None:
    """Each facet keeps Plotly's own fit to its data, which hides nothing."""
    figure = _grid_figure(grid)
    # Self-check: no facet holds a single repeated value, which gets a range.
    assert all(len(set(_plotted(figure, axis))) > 1 for axis in ("y", "y2", "y6"))
    assert all(axis.range is None for axis in _axes(figure, "yaxis"))


def test_a_facet_of_one_repeated_value_runs_from_zero(grid) -> None:
    """Plotly would autoscale it to a sliver around the value, with nonsense ticks."""
    late = "time_series_d"
    data = _grid(_snapshot({**SPANS, late: range(19, len(WEEKS))}))
    selections = [(late, FIRST_ORIGIN), *SIX[1:]]
    figure = _grid_figure(data, selections)
    values = _plotted(figure, "y")
    # Self-check: the late series has one whole month, the panel's last.
    assert len(values) == 1
    (facet,) = _axes(figure, "yaxis")[:1]
    assert facet.range[0] == 0
    assert facet.range[1] > values[0]
    assert all(axis.range is None for axis in _axes(figure, "yaxis")[1:])


def test_a_negative_flat_range_runs_up_to_zero() -> None:
    """Revenue can fall below zero, and the range must still ascend."""
    low, high = line_grid._y_range(pd.Series([-5.0, -5.0]))
    assert low < high
    assert low <= -5.0 <= high
    assert low <= 0.0 <= high


def test_a_window_of_zeros_fits_y_to_them_not_to_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Left to Plotly, a facet at 0 in view scales to its history out of view."""
    snapshot = _snapshot(SPANS)
    dormant = (snapshot["unique_id"] == "time_series_a") & (snapshot["ds"] >= WEEKS[3])
    snapshot.loc[dormant, "y"] = 0.0
    monkeypatch.setattr(line_grid, "_HISTORY", pd.Timedelta(weeks=4))
    data = _grid(snapshot)
    start = FIRST_ORIGIN - line_grid._HISTORY
    facet = line_grid.build_facet(
        data, "time_series_a", FIRST_ORIGIN, line_grid.MONTHLY
    )
    history = facet.actual.loc[facet.actual["ds"] < start, "y"].dropna()
    # Self-check: every value in view is 0, and the history out of view is not.
    assert set(line_grid._visible(facet, start)) == {0.0}
    assert history.min() > 0
    figure = _grid_figure(
        data, [("time_series_a", FIRST_ORIGIN), *SIX[1:]], window=True
    )
    (resolved,) = _axes(figure.full_figure_for_development(warn=False), "yaxis")[:1]
    assert resolved.range[1] < history.min()


def test_the_legend_names_each_trace_once(grid) -> None:
    """Six facets draw each series six times; the legend lists it once."""
    figure = _grid_figure(grid)
    shown = [trace.name for trace in figure.data if trace.showlegend]
    drawn = {trace.name for trace in figure.data if trace.name}
    assert sorted(shown) == sorted(set(shown))
    assert set(shown) == drawn


def test_one_model_in_both_roles_names_each_role() -> None:
    """Named alike, the legend drops the benchmark's line and the tooltips match."""
    data = _grid(_snapshot(SPANS), dict.fromkeys(line_grid.ROLES, CHALLENGER))
    table = data.per_series_comparison
    # Self-check: one model's forecast, scored in both roles.
    assert table["monthly_forecast_ch"].equals(table["monthly_forecast_bm"])
    shown = [trace for trace in _grid_figure(data).data if trace.showlegend]
    for role in line_grid.ROLES:
        for horizon in HORIZONS:
            (entry,) = [t for t in shown if t.legendgroup == f"{role} {horizon}"]
            assert f"({role})" in entry.name
            assert f"({role})" in entry.hovertemplate


def test_the_default_selection_is_the_first_series_at_the_last_origin(
    per_series,
) -> None:
    """Until a facet is chosen, every facet shows the run's latest forecast."""
    series, origin = line_grid.default_selection(per_series)
    assert series == min(per_series["unique_id"])
    assert origin == per_series["forecast_origin_date"].max()


# Unannotated: `AppTest.from_function` runs this function's source as a script of
# its own, where the annotations are evaluated and `line_grid` is undefined.
def _line_grid_page(data):
    """The two grids alone, as a script `AppTest` can run."""
    from dashboard.train import line_grid

    line_grid.render_line_grids(data)


def _actual_points(app, grid: int) -> int:
    """How many points grid `grid`'s first facet draws for its actual."""
    import json

    charts = app.get("plotly_chart")
    actual = next(
        trace
        for trace in json.loads(charts[grid - 1].proto.spec)["data"]
        if trace.get("name") == line_grid.ACTUAL
    )
    return len(actual["x"])


def test_each_grid_opens_monthly_independent_and_at_full_history(grid) -> None:
    """The defaults that hide nothing and match what every metric scores."""
    app = AppTest.from_function(_line_grid_page, args=(grid,)).run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    for index in range(1, line_grid.GRIDS + 1):
        assert app.radio(key=line_grid.grain_key(index)).value == line_grid.MONTHLY
        assert app.radio(key=line_grid.y_key(index)).value == "independent"
        assert app.radio(key=line_grid.window_key(index)).value == "full history"


def test_each_grid_takes_its_own_grain(grid) -> None:
    """Grid 2 can show the weeks under grid 1's months, same series, same origin."""
    app = AppTest.from_function(_line_grid_page, args=(grid,)).run()
    monthly = _actual_points(app, 1)
    app.radio(key=line_grid.grain_key(1)).set_value(line_grid.WEEKLY).run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    assert _actual_points(app, 1) > monthly
    assert _actual_points(app, 2) == monthly


# ================================================
# the selectors
# ================================================

FACETS = range(1, line_grid.ROWS * line_grid.COLS + 1)


def _selected(app, grid: int, facet: int) -> tuple:
    """One facet's series and origin as its selectors hold them."""
    return tuple(
        app.selectbox(key=line_grid.selection_key(grid, facet, kind)).value
        for kind in ("series", "origin")
    )


def _titles(app, grid: int) -> list[str]:
    """Grid `grid`'s facet titles, in facet order."""
    import json

    layout = json.loads(app.get("plotly_chart")[grid - 1].proto.spec)["layout"]
    return [
        annotation["text"]
        for annotation in layout["annotations"]
        if annotation.get("name") == line_grid.TITLE_ANNOTATION
    ]


def test_every_selector_opens_on_the_first_series_at_the_last_origin(grid) -> None:
    """All twelve facets of both grids, until one is chosen."""
    app = AppTest.from_function(_line_grid_page, args=(grid,)).run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    expected = line_grid.default_selection(grid.per_series_comparison)
    for index in range(1, line_grid.GRIDS + 1):
        assert all(_selected(app, index, facet) == expected for facet in FACETS)


def test_global_leads_the_series_and_the_latest_origin_leads_the_origins(grid) -> None:
    """Where 800 series cannot bury global, and the default origin is on top."""
    app = AppTest.from_function(_line_grid_page, args=(grid,)).run()
    series = app.selectbox(key=line_grid.selection_key(1, 1, "series")).options
    origins = app.selectbox(key=line_grid.selection_key(1, 1, "origin")).options
    assert series[0] == line_grid.GLOBAL
    assert origins == sorted(origins, reverse=True)
    # Self-check: the run has more than one origin, so the order is visible.
    assert len(origins) > 1


def test_each_facet_draws_its_own_selection(grid) -> None:
    """Facet 2's selector reaches facet 2 alone."""
    app = AppTest.from_function(_line_grid_page, args=(grid,)).run()
    app.selectbox(key=line_grid.selection_key(1, 2, "series")).set_value(
        "time_series_b"
    ).run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    titles = _titles(app, 1)
    # Self-check: the default series is another one.
    assert titles[0] != "time_series_b"
    assert titles[1] == "time_series_b"
    assert titles[2:] == [titles[0]] * (len(titles) - 2)


def test_the_copy_button_copies_facet_1_into_its_own_grid_only(grid) -> None:
    """Grid 1's other facets take facet 1's series and origin; grid 2 keeps its own."""
    app = AppTest.from_function(_line_grid_page, args=(grid,)).run()
    chosen = (line_grid.GLOBAL, FIRST_ORIGIN)
    for kind, value in zip(("series", "origin"), chosen, strict=True):
        app.selectbox(key=line_grid.selection_key(1, 1, kind)).set_value(value).run()
    untouched = _selected(app, 2, 1)
    # Self-check: the choice differs from the defaults the other facets hold.
    assert untouched != chosen
    app.button(key=line_grid.copy_key(1)).click().run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    assert all(_selected(app, 1, facet) == chosen for facet in FACETS)
    assert all(_selected(app, 2, facet) == untouched for facet in FACETS)


def test_globals_label_prints_its_series_at_the_chosen_origin(grid) -> None:
    """The fold set moves with the origin, and the label shows by how much."""
    app = AppTest.from_function(_line_grid_page, args=(grid,)).run()
    labels = {}
    for origin in (FIRST_ORIGIN, SECOND_ORIGIN):
        app.selectbox(key=line_grid.selection_key(1, 1, "series")).set_value(
            line_grid.GLOBAL
        ).run()
        app.selectbox(key=line_grid.selection_key(1, 1, "origin")).set_value(
            origin
        ).run()
        labels[origin] = _titles(app, 1)[0]
        size = len(line_grid.fold_set(grid.per_series_comparison, origin))
        assert f"({size} series at this origin)" in labels[origin]
    # Self-check: the two origins' fold sets differ, so N had to move.
    assert len(set(labels.values())) == len(labels)


def test_a_selection_this_run_lacks_falls_back_to_the_default(grid) -> None:
    """Session state outlives a run switch; an origin the new run lacks would raise."""
    app = AppTest.from_function(_line_grid_page, args=(grid,))
    app.session_state[line_grid.selection_key(1, 1, "origin")] = pd.Timestamp(
        "1999-01-03"
    )
    app.session_state[line_grid.selection_key(1, 1, "series")] = "time_series_z"
    app.run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    assert _selected(app, 1, 1) == line_grid.default_selection(
        grid.per_series_comparison
    )
