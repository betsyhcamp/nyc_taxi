"""Residual diagnostics plotting utilities for time series forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.figure as mpl_fig
    import plotly.graph_objects as go
    import polars as pl

    DataFrameLike: TypeAlias = pd.DataFrame | pl.DataFrame
    FigureLike: TypeAlias = mpl_fig.Figure | go.Figure
else:
    DataFrameLike = pd.DataFrame
    FigureLike = object


_Z_SCORE_95 = 1.96
_COLOR_PRIMARY = "#1f77b4"
_COLOR_SECONDARY = "#ff7f0e"
_DPI = 100
_LINEWIDTH_PRIMARY = 1


@dataclass(frozen=True)
class ResidualDiagnostics:
    """Precomputed diagnostic data for residual analysis.

    Separates data preparation for rendering, enabling:
    - Reuse of computed statistics without replotting
    - Simpler, more testable plotting functions
    - clear contract between computation and visualization
    """

    timestamps: np.ndarray
    actual: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    acf_values: np.ndarray
    conf_interval: float
    kde_x: np.ndarray
    kde_y: np.ndarray


def _compute_diagnostics(
    df: pd.DataFrame,
    time_col: str,
    actual_col: str,
    fitted_col: str,
    nlags: int | None,
) -> ResidualDiagnostics:
    """Compute all diagnostic values from input data."""
    from scipy.stats import gaussian_kde
    from statsmodels.tsa.stattools import acf

    # Sort by time
    df = df.sort_values(by=time_col).reset_index(drop=True)

    # Extract data
    timestamps = df[time_col].to_numpy()
    actual = df[actual_col].to_numpy(dtype=float)
    fitted = df[fitted_col].to_numpy(dtype=float)
    residuals = actual - fitted

    # Compute ACF
    n = len(residuals)
    if nlags is None:
        nlags = max(1, min(40, n // 4))  # Guard against n<4
    acf_result = acf(
        residuals, nlags=nlags, fft=True
    )  # result return type depends on args
    acf_values: np.ndarray = (
        acf_result[0] if isinstance(acf_result, tuple) else acf_result
    )
    conf_interval = _Z_SCORE_95 / np.sqrt(
        n
    )  # assume two sided 95%CI, normal distribution

    # Compute KDE
    if np.isclose(np.std(residuals), 0, atol=1e-10):
        raise ValueError("Residuals have near-zero variance; KDE is undefined.")
    kde = gaussian_kde(residuals)
    kde_x = np.linspace(residuals.min(), residuals.max(), 200)
    kde_y = kde(kde_x)

    return ResidualDiagnostics(
        timestamps=timestamps,
        actual=actual,
        fitted=fitted,
        residuals=residuals,
        acf_values=acf_values,
        conf_interval=conf_interval,
        kde_x=kde_x,
        kde_y=kde_y,
    )


def plot_residual_diagnostics(
    df: DataFrameLike,
    time_col: str,
    actual_col: str,
    fitted_col: str,
    backend: Literal["plotly", "matplotlib"] = "matplotlib",
    width: int = 1200,
    height: int = 800,
    nlags: int | None = None,
    hist_bins: int | str = "auto",
    return_fig: bool = False,
) -> FigureLike | None:
    """Plot residual diagnostics for a single time series forecasting model.

    Generates a 4-panel diagnostic plot:
    - Actual vs Fitted values over time
    - Residuals over time
    - ACF of residuals
    - Histogram of residuals with KDE overlay

    Args:
        df: DataFrame containing time series data (pandas or polars).
        time_col: Column name for time index.
        actual_col: Column name for actual values.
        fitted_col: Column name for fitted values.
        backend: Plotting backend, either "plotly" or "matplotlib".
        width: Figure width in pixels.
        height: Figure height in pixels.
        nlags: Number of lags for ACF. If None, uses min(40, n//4).
        hist_bins: Number of bins for histogram, or "auto".
        return_fig: If True, return the figure object instead of displaying.

    Returns:
        None if return_fig=False, otherwise the figure object.

    Raises:
        ValueError: If required columns are missing, contain NaN values,
            or if invalid parameters are provided.
    """
    # Convert polars to pandas if needed
    df = _convert_to_pandas(df)

    # Validate inputs
    _validate_inputs(df, time_col, actual_col, fitted_col, backend, width, height)

    diagnostics_data = _compute_diagnostics(df, time_col, actual_col, fitted_col, nlags)

    # Dispatch to backend
    if backend == "matplotlib":
        fig = _plot_matplotlib(
            diagnostics_data,
            hist_bins,
            width,
            height,
        )
        if return_fig:
            return fig
        import matplotlib.pyplot as plt

        plt.show()
        return None
    else:
        fig = _plot_plotly(
            diagnostics_data,
            hist_bins,
            width,
            height,
        )
        if return_fig:
            return fig
        fig.show()
        return None


def _convert_to_pandas(df: DataFrameLike) -> pd.DataFrame:
    """Convert polars DataFrame to pandas if needed."""
    if isinstance(df, pd.DataFrame):
        return df
    return df.to_pandas()


def _validate_inputs(
    df: pd.DataFrame,
    time_col: str,
    actual_col: str,
    fitted_col: str,
    backend: str,
    width: int,
    height: int,
) -> None:
    """Validate input parameters."""

    if len(df) < 2:
        raise ValueError(f"DataFrame must have at least 2 rows, got {len(df)}.")

    cols_list = [time_col, actual_col, fitted_col]

    # Check columns exist
    missing_cols_list = [col for col in cols_list if col not in df.columns]

    if missing_cols_list:
        raise ValueError(
            f"Column(s) {missing_cols_list} not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Check for NaN values
    na_counts = df[cols_list].isna().sum()
    has_na = na_counts[na_counts > 0]
    if len(has_na) > 0:
        details = ", ".join(f"{col}={int(count)}" for col, count in has_na.items())
        raise ValueError(f"NaN values found (column=count): {details}")

    # Check backend
    if backend not in ("plotly", "matplotlib"):
        raise ValueError(
            f"Invalid backend '{backend}'. Must be 'plotly' or 'matplotlib'."
        )

    # Check dimensions
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}.")
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}.")


def _plot_matplotlib(
    data: ResidualDiagnostics,
    hist_bins: int | str,
    width: int,
    height: int,
) -> mpl_fig.Figure:
    """Create diagnostic plot using matplotlib."""
    import matplotlib.pyplot as plt

    figsize = (width / _DPI, height / _DPI)

    fig = plt.figure(figsize=figsize, dpi=_DPI)

    # Layout: 3 rows, 2 columns
    # Row 1: Actual vs Fitted (spans both columns)
    # Row 2: Residuals (spans both columns)
    # Row 3: ACF (left), Histogram (right)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.25)

    # Panel 1: Actual vs Fitted
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(
        data.timestamps,
        data.actual,
        label="Actual",
        color=_COLOR_PRIMARY,
        linewidth=_LINEWIDTH_PRIMARY,
    )
    ax1.plot(
        data.timestamps,
        data.fitted,
        label="Fitted",
        color=_COLOR_SECONDARY,
        linewidth=_LINEWIDTH_PRIMARY,
    )
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax1.set_ylabel("Value")
    ax1.grid(alpha=0.3)

    # Panel 2: Residuals over time
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax2.plot(
        data.timestamps,
        data.residuals,
        color=_COLOR_PRIMARY,
        linewidth=_LINEWIDTH_PRIMARY,
    )
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Time")
    ax2.grid(alpha=0.3)

    # Hide x-axis labels on top panel (shared x-axis)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Panel 3: ACF
    ax3 = fig.add_subplot(gs[2, 0])
    lags = np.arange(len(data.acf_values))
    ax3.vlines(
        lags, 0, data.acf_values, color=_COLOR_PRIMARY, linewidth=_LINEWIDTH_PRIMARY
    )
    ax3.scatter(lags, data.acf_values, color=_COLOR_PRIMARY, s=10, zorder=3)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax3.axhline(y=data.conf_interval, color="red", linestyle="--", linewidth=0.8)
    ax3.axhline(y=-data.conf_interval, color="red", linestyle="--", linewidth=0.8)
    ax3.set_xlabel("Lag")
    ax3.set_ylabel("ACF")
    ax3.grid(alpha=0.3)

    # Panel 4: Histogram with KDE
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(
        data.residuals,
        bins=hist_bins,
        density=True,
        alpha=0.7,
        color=_COLOR_PRIMARY,
        edgecolor="white",
    )
    ax4.plot(data.kde_x, data.kde_y, color=_COLOR_SECONDARY, linewidth=1.5)
    ax4.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax4.axvline(x=data.residuals.mean(), color="red", linestyle="--", linewidth=0.8)
    ax4.set_xlabel("Residual")
    ax4.set_ylabel("Density")
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def _plot_plotly(
    data: ResidualDiagnostics,
    hist_bins: int | str,
    width: int,
    height: int,
) -> go.Figure:
    """Create diagnostic plot using plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3,
        cols=2,
        row_heights=[0.33, 0.33, 0.33],
        column_widths=[0.5, 0.5],
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{}, {}],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        shared_xaxes=True,
    )

    # Panel 1: Actual vs Fitted
    fig.add_trace(
        go.Scatter(
            x=data.timestamps,
            y=data.actual,
            name="Actual",
            line=dict(color=_COLOR_PRIMARY, width=_LINEWIDTH_PRIMARY),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.timestamps,
            y=data.fitted,
            name="Fitted",
            line=dict(color=_COLOR_SECONDARY, width=_LINEWIDTH_PRIMARY),
        ),
        row=1,
        col=1,
    )

    # Panel 2: Residuals over time
    fig.add_trace(
        go.Scatter(
            x=data.timestamps,
            y=data.residuals,
            name="Residuals",
            line=dict(color=_COLOR_PRIMARY, width=_LINEWIDTH_PRIMARY),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color="black", width=_LINEWIDTH_PRIMARY), row=2, col=1)

    # Panel 3: ACF
    lags = list(range(len(data.acf_values)))
    for lag, val in zip(lags, data.acf_values):
        fig.add_trace(
            go.Scatter(
                x=[lag, lag],
                y=[0, val],
                mode="lines",
                line=dict(color=_COLOR_PRIMARY, width=_LINEWIDTH_PRIMARY),
                showlegend=False,
            ),
            row=3,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=data.acf_values,
            mode="markers",
            marker=dict(color=_COLOR_PRIMARY, size=5),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color="black", width=_LINEWIDTH_PRIMARY), row=3, col=1)
    fig.add_hline(
        y=data.conf_interval,
        line=dict(color="red", width=_LINEWIDTH_PRIMARY, dash="dash"),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=-data.conf_interval,
        line=dict(color="red", width=_LINEWIDTH_PRIMARY, dash="dash"),
        row=3,
        col=1,
    )

    # Panel 4: Histogram with KDE
    nbins = hist_bins if isinstance(hist_bins, int) else None
    fig.add_trace(
        go.Histogram(
            x=data.residuals,
            nbinsx=nbins,
            histnorm="probability density",
            marker=dict(
                color=_COLOR_PRIMARY, line=dict(color="white", width=_LINEWIDTH_PRIMARY)
            ),
            opacity=0.7,
            showlegend=False,
        ),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=data.kde_x,
            y=data.kde_y,
            mode="lines",
            line=dict(color=_COLOR_SECONDARY, width=1.5),
            showlegend=False,
        ),
        row=3,
        col=2,
    )
    fig.add_vline(x=0, line=dict(color="black", width=_LINEWIDTH_PRIMARY), row=3, col=2)
    fig.add_vline(
        x=float(data.residuals.mean()),
        line=dict(color="red", width=_LINEWIDTH_PRIMARY, dash="dash"),
        row=3,
        col=2,
    )

    # Update layout
    fig.update_layout(
        width=width,
        height=height,
        showlegend=True,
        legend=dict(x=1.01, y=1, xanchor="left", yanchor="top"),
        margin=dict(l=60, r=100, t=40, b=60),
    )

    # Axis labels
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Lag", row=3, col=1)
    fig.update_yaxes(title_text="ACF", row=3, col=1)
    fig.update_xaxes(title_text="Residual", row=3, col=2)
    fig.update_yaxes(title_text="Density", row=3, col=2)

    return fig
