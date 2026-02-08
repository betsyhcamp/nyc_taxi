"""Residual diagnostics plotting utilities for time series forecasting."""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import acf


def plot_residual_diagnostics(
    df: Union[pd.DataFrame, "pl.DataFrame"],
    time_col: str,
    actual_col: str,
    fitted_col: str,
    backend: Literal["plotly", "matplotlib"] = "matplotlib",
    width: int = 1200,
    height: int = 800,
    nlags: int | None = None,
    hist_bins: Union[int, str] = "auto",
    return_fig: bool = False,
) -> Union[None, "plt.Figure", "go.Figure"]:
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

    # Sort by time
    df = df.sort_values(by=time_col).reset_index(drop=True)

    # Extract data
    time_vals = df[time_col].values
    actual = df[actual_col].values
    fitted = df[fitted_col].values
    residuals = actual - fitted

    # Compute ACF
    n = len(residuals)
    if nlags is None:
        nlags = min(40, n // 4)
    acf_values = acf(residuals, nlags=nlags, fft=True)
    conf_interval = 1.96 / np.sqrt(n)

    # Compute KDE
    kde = gaussian_kde(residuals)
    kde_x = np.linspace(residuals.min(), residuals.max(), 200)
    kde_y = kde(kde_x)

    # Dispatch to backend
    if backend == "matplotlib":
        fig = _plot_matplotlib(
            time_vals,
            actual,
            fitted,
            residuals,
            acf_values,
            nlags,
            conf_interval,
            kde_x,
            kde_y,
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
            time_vals,
            actual,
            fitted,
            residuals,
            acf_values,
            nlags,
            conf_interval,
            kde_x,
            kde_y,
            hist_bins,
            width,
            height,
        )
        if return_fig:
            return fig
        fig.show()
        return None


def _convert_to_pandas(df: Union[pd.DataFrame, "pl.DataFrame"]) -> pd.DataFrame:
    """Convert polars DataFrame to pandas if needed."""
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return df


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
    # Check columns exist
    missing_cols = []
    for col in [time_col, actual_col, fitted_col]:
        if col not in df.columns:
            missing_cols.append(col)
    if missing_cols:
        raise ValueError(
            f"Column(s) {missing_cols} not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Check for NaN values
    for col in [time_col, actual_col, fitted_col]:
        if df[col].isna().any():
            raise ValueError(
                f"Column '{col}' contains missing values. "
                f"Found {df[col].isna().sum()} NaN values."
            )

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
    time_vals: np.ndarray,
    actual: np.ndarray,
    fitted: np.ndarray,
    residuals: np.ndarray,
    acf_values: np.ndarray,
    nlags: int,
    conf_interval: float,
    kde_x: np.ndarray,
    kde_y: np.ndarray,
    hist_bins: Union[int, str],
    width: int,
    height: int,
) -> "plt.Figure":
    """Create diagnostic plot using matplotlib."""
    import matplotlib.pyplot as plt

    DPI = 100
    figsize = (width / DPI, height / DPI)

    fig = plt.figure(figsize=figsize, dpi=DPI)

    # Layout: 3 rows, 2 columns
    # Row 1: Actual vs Fitted (spans both columns)
    # Row 2: Residuals (spans both columns)
    # Row 3: ACF (left), Histogram (right)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.25)

    # Panel 1: Actual vs Fitted
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_vals, actual, label="Actual", color="#1f77b4", linewidth=1)
    ax1.plot(time_vals, fitted, label="Fitted", color="#ff7f0e", linewidth=1)
    ax1.legend(loc="upper right")
    ax1.set_ylabel("Value")
    ax1.grid(alpha=0.3)

    # Panel 2: Residuals over time
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax2.plot(time_vals, residuals, color="#1f77b4", linewidth=1)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Time")
    ax2.grid(alpha=0.3)

    # Hide x-axis labels on top panel (shared x-axis)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Panel 3: ACF
    ax3 = fig.add_subplot(gs[2, 0])
    lags = np.arange(len(acf_values))
    ax3.vlines(lags, 0, acf_values, color="#1f77b4", linewidth=1)
    ax3.scatter(lags, acf_values, color="#1f77b4", s=10, zorder=3)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax3.axhline(y=conf_interval, color="red", linestyle="--", linewidth=0.8)
    ax3.axhline(y=-conf_interval, color="red", linestyle="--", linewidth=0.8)
    ax3.set_xlabel("Lag")
    ax3.set_ylabel("ACF")
    ax3.grid(alpha=0.3)

    # Panel 4: Histogram with KDE
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(
        residuals,
        bins=hist_bins,
        density=True,
        alpha=0.7,
        color="#1f77b4",
        edgecolor="white",
    )
    ax4.plot(kde_x, kde_y, color="#ff7f0e", linewidth=1.5)
    ax4.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax4.axvline(x=residuals.mean(), color="red", linestyle="--", linewidth=0.8)
    ax4.set_xlabel("Residual")
    ax4.set_ylabel("Density")
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def _plot_plotly(
    time_vals: np.ndarray,
    actual: np.ndarray,
    fitted: np.ndarray,
    residuals: np.ndarray,
    acf_values: np.ndarray,
    nlags: int,
    conf_interval: float,
    kde_x: np.ndarray,
    kde_y: np.ndarray,
    hist_bins: Union[int, str],
    width: int,
    height: int,
) -> "go.Figure":
    """Create diagnostic plot using plotly."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

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
            x=time_vals, y=actual, name="Actual", line=dict(color="#1f77b4", width=1)
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_vals, y=fitted, name="Fitted", line=dict(color="#ff7f0e", width=1)
        ),
        row=1,
        col=1,
    )

    # Panel 2: Residuals over time
    fig.add_trace(
        go.Scatter(
            x=time_vals,
            y=residuals,
            name="Residuals",
            line=dict(color="#1f77b4", width=1),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color="black", width=1), row=2, col=1)

    # Panel 3: ACF
    lags = list(range(len(acf_values)))
    for i, (lag, val) in enumerate(zip(lags, acf_values)):
        fig.add_trace(
            go.Scatter(
                x=[lag, lag],
                y=[0, val],
                mode="lines",
                line=dict(color="#1f77b4", width=1),
                showlegend=False,
            ),
            row=3,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=acf_values,
            mode="markers",
            marker=dict(color="#1f77b4", size=5),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color="black", width=1), row=3, col=1)
    fig.add_hline(
        y=conf_interval, line=dict(color="red", width=1, dash="dash"), row=3, col=1
    )
    fig.add_hline(
        y=-conf_interval, line=dict(color="red", width=1, dash="dash"), row=3, col=1
    )

    # Panel 4: Histogram with KDE
    nbins = hist_bins if isinstance(hist_bins, int) else None
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=nbins,
            histnorm="probability density",
            marker=dict(color="#1f77b4", line=dict(color="white", width=1)),
            opacity=0.7,
            showlegend=False,
        ),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=kde_x,
            y=kde_y,
            mode="lines",
            line=dict(color="#ff7f0e", width=1.5),
            showlegend=False,
        ),
        row=3,
        col=2,
    )
    fig.add_vline(x=0, line=dict(color="black", width=1), row=3, col=2)
    fig.add_vline(
        x=float(residuals.mean()),
        line=dict(color="red", width=1, dash="dash"),
        row=3,
        col=2,
    )

    # Update layout
    fig.update_layout(
        width=width,
        height=height,
        showlegend=True,
        legend=dict(x=0.85, y=0.98),
        margin=dict(l=60, r=40, t=40, b=60),
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
