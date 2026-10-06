"""Every color, marker and per-metric rule the backtest monitor draws with.

Plotly interpolates Viridis from ten stops and matplotlib from 256, so a live
sample differs by one or two RGB units per library. Hence tier literals, and
CLIP_* as provenance: editing one fails the derivation test, it does not recolor.

TIER_SYMBOL is load-bearing, not decoration.
"""

ALPHA = 0.85

PLOT_BGCOLOR = "#ffffff"
"""Plotly's default plot area is grey, so set this and `paper_bgcolor`."""

FONT_COLOR = "#2b2b2b"
AXIS_COLOR = "#6f6e6b"

# Draw the grid with `layer="below traces"`. Plotly defaults to above, where a
# translucent line tints every mark it crosses.
GRID_COLOR = AXIS_COLOR
GRID_ALPHA = 0.18

# The light end binds: it needs 0.23 on white opaque, 0.30 at ALPHA, to clear the
# 2.0:1 ordinal floor. Clipping the dark end costs separation for nothing.
CLIP_DARK = 0.00
CLIP_LIGHT = 0.35

METRICS = ("wrmae_pooled", "wape", "signed_bias_pooled")
"""The three the dashboard displays. The evaluate outputs carry six."""

TIERS = ("very_low", "low", "middle", "high", "very_high")
"""The ramp's domain. A tier grid reads the run's configured labels instead."""

# A guardrail breach is direction from the baseline, never a recolored mark,
# which would lose horizon identity exactly where it failed.
HORIZON_COLOR = {"horizon_1": "#2a78d6", "horizon_2": "#d95926"}
HORIZON_MARKER = {"horizon_1": "circle", "horizon_2": "diamond"}
ACTUAL_INK = "#52514e"

TIER_COLOR = {
    "very_low": "#440154",
    "low": "#453882",
    "middle": "#31668e",
    "high": "#228d8d",
    "very_high": "#2fb47c",
}

TIER_SYMBOL = {
    "very_low": "square-open",
    "low": "triangle-up-open",
    "middle": "triangle-down-open",
    "high": "x-thin",
    "very_high": "cross-thin",
}

# Drawn as a constant, never read off a benchmark row: that row reads near 1.0,
# not 1.0, because `wrmae_per_series` renormalizes.
REFERENCE_LINE = {"wrmae_pooled": 1.0, "wape": None, "signed_bias_pooled": 0.0}
SHOW_BENCHMARK = {"wrmae_pooled": False, "wape": True, "signed_bias_pooled": True}

# Skill is one-sided; a symmetric axis would spend most of itself unvisited.
SYMMETRIC_RANGE = {
    "wrmae_pooled": False,
    "wape": False,
    "signed_bias_pooled": True,
}


def rgba(hex_color: str, alpha: float = ALPHA) -> str:
    """Plotly's `rgba()` string, since a hex literal carries no alpha channel."""
    raw = hex_color.lstrip("#")
    red, green, blue = (int(raw[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"
