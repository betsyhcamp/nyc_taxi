"""Every color, marker and per-metric rule the backtest monitor draws with.

Plotly interpolates Viridis from ten stops and matplotlib from 256, so a live
sample differs by one or two RGB units per library. Hence tier literals, and
CLIP_* as provenance: editing one fails the derivation test, it does not recolor.

TIER_SYMBOLS is load-bearing, not decoration.
"""

from collections.abc import Sequence

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

# A guardrail breach is direction from the baseline, never a recolored mark,
# which would lose horizon identity exactly where it failed.
HORIZON_COLOR = {"horizon_1": "#2a78d6", "horizon_2": "#d95926"}
HORIZON_MARKER = {"horizon_1": "circle", "horizon_2": "diamond"}
ACTUAL_INK = "#52514e"

TIER_RAMPS = {
    2: ("#440154", "#2fb47c"),
    3: ("#440154", "#31668e", "#2fb47c"),
    4: ("#440154", "#3f4889", "#27808e", "#2fb47c"),
    5: ("#440154", "#453882", "#31668e", "#228d8d", "#2fb47c"),
}
"""Viridis resampled over the clip at each tier count, lowest tier first. Not a
subset of one ramp, which at four tiers steps less far. Ramps pass the gates to
seven; nothing above five, which would need symbols no one has measured."""

TIER_SYMBOLS = (
    "square-open",
    "triangle-up-open",
    "triangle-down-open",
    "x-thin",
    "cross-thin",
)
"""In tier order, n tiers taking the first n. Shape carries identity: a ramp is
validated on adjacent steps, while a scatter puts any two tiers side by side."""

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


def tier_encoding(tier_labels: Sequence[str]) -> dict[str, tuple[str, str]]:
    """Each configured tier's color and symbol, by rank.

    By the configured count, not the count present: a run whose `assign_tiers`
    emitted a prefix keeps its config's ramp, and a prefix of validated adjacent
    steps is itself validated.

    Args:
        tier_labels: The run's configured labels, lowest tier first.

    Raises:
        ValueError: If no ramp exists for that many tiers.

    Returns:
        dict[str, tuple[str, str]]: Each label to its hex color and Plotly symbol.
    """
    ramp = TIER_RAMPS.get(len(tier_labels))
    if ramp is None:
        raise ValueError(
            f"{len(tier_labels)} configured tiers, while tier ramps are validated "
            f"for {min(TIER_RAMPS)} to {max(TIER_RAMPS)} only."
        )
    symbols = TIER_SYMBOLS[: len(ramp)]
    return {
        label: (color, symbol)
        for label, color, symbol in zip(tier_labels, ramp, symbols, strict=True)
    }


def rgba(hex_color: str, alpha: float = ALPHA) -> str:
    """Plotly's `rgba()` string, since a hex literal carries no alpha channel."""
    raw = hex_color.lstrip("#")
    red, green, blue = (int(raw[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"
