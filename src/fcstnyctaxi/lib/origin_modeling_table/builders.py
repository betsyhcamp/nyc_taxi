from typing import cast

import pandas as pd
from mlforecast import MLForecast


def _require_columns(df: pd.DataFrame, required: list[str], frame_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def trim_incomplete_series_months(
    panel_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop (series, month) pairs that lack all weeks_in_month fiscal weeks.

    Framing-specific cleaning for direct-month targets: a full month target and
    weeks_actualized indexing both presume every week of the month is present.

    Args:
        panel_df: Weekly panel with columns (unique_id, ds, y).
        calendar_df: Fiscal calendar with columns (ds, fiscal_year_month,
            fiscal_week_of_month, weeks_in_month).

    Returns:
        (trimmed_panel_df, dropped_pairs). trimmed_panel_df carries panel_df's
        three columns. dropped_pairs has one row per dropped (series, month),
        with columns (unique_id, fiscal_year_month, weeks_present,
        weeks_in_month).

    Raises:
        ValueError: If either frame lacks a required column; if calendar_df does
            not cover every panel date, since an unlabeled row would otherwise
            be dropped silently and never appear in dropped_pairs; or if a
            partial pair survives the trim.
    """
    _require_columns(
        df=panel_df, required=["unique_id", "ds", "y"], frame_name="panel_df"
    )
    _require_columns(
        df=calendar_df,
        required=["ds", "fiscal_year_month", "fiscal_week_of_month", "weeks_in_month"],
        frame_name="calendar_df",
    )

    # precondition: the calendar must cover every panel date
    uncovered = panel_df.loc[~panel_df["ds"].isin(calendar_df["ds"]), "ds"].unique()
    if len(uncovered):
        raise ValueError(
            f"calendar_df does not cover {len(uncovered)} panel date(s); "
            f"first few: {sorted(uncovered)[:5]}"
        )

    labeled = panel_df.merge(
        calendar_df[
            ["ds", "fiscal_year_month", "fiscal_week_of_month", "weeks_in_month"]
        ],
        on="ds",
        how="left",
    )
    weeks_present = labeled.groupby(["unique_id", "fiscal_year_month"])[
        "fiscal_week_of_month"
    ].transform("nunique")
    keep_row_mask = weeks_present == labeled["weeks_in_month"]
    trimmed_panel_df = labeled.loc[keep_row_mask, ["unique_id", "ds", "y"]]
    dropped_pairs = (
        labeled.loc[~keep_row_mask]
        .groupby(["unique_id", "fiscal_year_month"], as_index=False)
        .agg(
            weeks_present=("fiscal_week_of_month", "nunique"),
            weeks_in_month=("weeks_in_month", "first"),
        )
    )

    # check that we have correctly trimmed months that were incomplete
    check = trimmed_panel_df.merge(
        calendar_df[
            ["ds", "fiscal_year_month", "fiscal_week_of_month", "weeks_in_month"]
        ],
        on="ds",
        how="left",
    )
    weeks_present = check.groupby(["unique_id", "fiscal_year_month"])[
        "fiscal_week_of_month"
    ].transform("nunique")
    survivors = check.loc[weeks_present != check["weeks_in_month"]]
    if not survivors.empty:
        raise ValueError("trimmed panel still has partial (series, month) pairs")

    return trimmed_panel_df, dropped_pairs


def build_weekly_features(
    panel: pd.DataFrame,
    freq: str,
    *,
    lags: list[int],
    lag_transforms: dict[int, list],
) -> pd.DataFrame:
    """Build Layer 1 of the modeling table: weekly lag features per series.

    A feature factory, not a model — MLForecast is built with models=[] and
    only preprocess() is called. dropna=False is load-bearing: it keeps each
    series' early weeks, where lags are still NaN, so Layer 3's left join
    finds a row for every origin. Native MLForecast feature names are kept;
    the framing declares the same names in MLF_FEATURES and
    assert_preprocess_feature_drift gates the agreement.

    Args:
        panel: Weekly panel (requiresunique_id, ds, y); should be the trimmed panel
            from trim_incomplete_series_months.
        freq: Pandas offset alias for the series frequency, e.g. "W-SUN".
        lags: Lag periods to emit, in units of freq.
        lag_transforms: Lag period -> transforms applied at that lag, e.g.
            {1: [RollingMean(window_size=4)]}.

    Returns:
        DataFrame with (unique_id, feature_ds, y, + one column per native
        MLForecast feature name), one row per panel row. ds is renamed
        feature_ds, the join key Layer 3 uses.
    """
    mlf = MLForecast(models=[], freq=freq, lags=lags, lag_transforms=lag_transforms)
    feats = cast(
        pd.DataFrame, mlf.preprocess(panel, dropna=False)
    )  # unique_id, ds, y, + native feature names ; cast() to satisfy type checking
    return feats.rename(columns={"ds": "feature_ds"})
