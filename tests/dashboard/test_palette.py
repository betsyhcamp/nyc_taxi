from itertools import pairwise

import matplotlib
import pytest
from matplotlib.colors import to_hex

from dashboard.shared import palette

# WCAG for the first two. 2.0 is the ordinal-ramp convention, 0.06 an OKLab L
# step, both against the plot area.
MARK_CONTRAST_GATE = 3.0
TEXT_CONTRAST_GATE = 4.5
ORDINAL_CONTRAST_GATE = 2.0
ADJACENT_LIGHTNESS_GATE = 0.06

# Chrome has no WCAG gate. Conventional gridlines run 10% to 15% ink on white,
# measuring 1.25:1 to 1.41:1, so 1.15 sits just below the band.
GRID_CONTRAST_FLOOR = 1.15


def _channels(hex_color: str) -> tuple[int, int, int]:
    """The three 0-255 channels of a #rrggbb string."""
    raw = hex_color.lstrip("#")
    return tuple(int(raw[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _linear(hex_color: str) -> tuple[float, float, float]:
    """Linear-light RGB, the space both luminance and OKLab are defined over."""
    srgb = [channel / 255 for channel in _channels(hex_color)]
    return tuple(  # type: ignore[return-value]
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb
    )


def _relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance."""
    red, green, blue = _linear(hex_color)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(hex_a: str, hex_b: str) -> float:
    """WCAG contrast ratio between two #rrggbb colors."""
    high, low = sorted(
        (_relative_luminance(hex_a), _relative_luminance(hex_b)), reverse=True
    )
    return (high + 0.05) / (low + 0.05)


def composite_over(hex_fg: str, hex_bg: str, alpha: float) -> str:
    """The opaque color a translucent mark actually renders as."""
    blended = (
        round(alpha * fg + (1 - alpha) * bg)
        for fg, bg in zip(_channels(hex_fg), _channels(hex_bg), strict=True)
    )
    return "#" + "".join(f"{channel:02x}" for channel in blended)


def mark_contrast(hex_color: str, alpha: float = palette.ALPHA) -> float:
    """Contrast of something drawn at `alpha` over the plot area."""
    composited = composite_over(hex_color, palette.PLOT_BGCOLOR, alpha)
    return contrast_ratio(composited, palette.PLOT_BGCOLOR)


def oklab_lightness(hex_color: str) -> float:
    """OKLab L, which is perceptually uniform where WCAG luminance is not."""
    red, green, blue = _linear(hex_color)
    long_ = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    medium = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    short = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    cones = [channel ** (1 / 3) for channel in (long_, medium, short)]
    return 0.2104542553 * cones[0] + 0.7936177850 * cones[1] - 0.0040720468 * cones[2]


# ================================================
# the color math itself
# ================================================


def test_contrast_ratio_hits_its_known_anchors() -> None:
    """Self-check on the calculator, since broken math reads as a broken palette."""
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, abs=1e-9)
    assert contrast_ratio("#777777", "#777777") == pytest.approx(1.0, abs=1e-9)


def test_composite_over_spans_both_ends_of_alpha() -> None:
    """Self-check: alpha 1 keeps the mark, alpha 0 leaves the ground."""
    assert composite_over("#2a78d6", "#ffffff", 1.0) == "#2a78d6"
    assert composite_over("#2a78d6", "#ffffff", 0.0) == "#ffffff"


def test_rgba_preserves_the_channels_and_carries_the_alpha() -> None:
    """A transposed channel in `rgba` would silently recolor every mark."""
    red, green, blue = _channels(palette.HORIZON_COLOR["horizon_1"])
    assert palette.rgba(palette.HORIZON_COLOR["horizon_1"]) == (
        f"rgba({red},{green},{blue},{palette.ALPHA})"
    )


# ================================================
# contrast and separation gates
# ================================================


def test_every_data_mark_clears_the_graphical_mark_gate() -> None:
    """The two horizon colors and the actual's ink, at alpha, on the plot area."""
    marks = {**palette.HORIZON_COLOR, "actual": palette.ACTUAL_INK}
    under_gate = {
        name: mark_contrast(color)
        for name, color in marks.items()
        if mark_contrast(color) < MARK_CONTRAST_GATE
    }
    assert not under_gate


def test_the_axis_color_clears_the_graphical_mark_gate() -> None:
    """The reference line is drawn as a constant in it, so it is a mark too."""
    assert contrast_ratio(palette.AXIS_COLOR, palette.PLOT_BGCOLOR) >= (
        MARK_CONTRAST_GATE
    )


def test_the_grid_is_visible_against_the_plot_area() -> None:
    """The grid composites over the plot area alone, below the traces."""
    assert mark_contrast(palette.GRID_COLOR, palette.GRID_ALPHA) >= GRID_CONTRAST_FLOOR


def test_the_grid_never_outweighs_the_faintest_datum() -> None:
    """Chrome stays under every data mark, the lightest tier included."""
    data_marks = (
        *palette.HORIZON_COLOR.values(),
        *palette.TIER_COLOR.values(),
        palette.ACTUAL_INK,
    )
    faintest = min(mark_contrast(color) for color in data_marks)
    assert mark_contrast(palette.GRID_COLOR, palette.GRID_ALPHA) < faintest


def test_the_font_color_clears_the_text_gate() -> None:
    """Text is drawn opaque, so it is measured opaque."""
    assert contrast_ratio(palette.FONT_COLOR, palette.PLOT_BGCOLOR) >= (
        TEXT_CONTRAST_GATE
    )


def test_the_tier_ramp_steps_far_enough_to_read_as_ordered() -> None:
    """Adjacent OKLab lightness across the ramp, in configured tier order."""
    lightness = [oklab_lightness(palette.TIER_COLOR[tier]) for tier in palette.TIERS]
    steps = [later - earlier for earlier, later in pairwise(lightness)]
    assert min(steps) >= ADJACENT_LIGHTNESS_GATE


def test_the_tier_ramp_rises_monotonically() -> None:
    """Lightness carries tier order, so a reordered literal must fail."""
    lightness = [oklab_lightness(palette.TIER_COLOR[tier]) for tier in palette.TIERS]
    assert lightness == sorted(lightness)


def test_the_lightest_tier_clears_the_ordinal_contrast_floor() -> None:
    """The step nearest the plot area is the binding one; CLIP_LIGHT buys it."""
    nearest = min(mark_contrast(palette.TIER_COLOR[tier]) for tier in palette.TIERS)
    assert nearest >= ORDINAL_CONTRAST_GATE


def test_tier_color_literals_match_the_colormap_at_the_documented_clip() -> None:
    """The literals exist so Plotly cannot resample; this is what they came from."""
    viridis = matplotlib.colormaps["viridis"]
    low, high = palette.CLIP_DARK, 1.0 - palette.CLIP_LIGHT
    step = (high - low) / (len(palette.TIERS) - 1)
    derived = {
        tier: to_hex(viridis(low + index * step))
        for index, tier in enumerate(palette.TIERS)
    }
    assert palette.TIER_COLOR == derived


# ================================================
# the two encoding invariants
# ================================================


def test_no_other_mark_reuses_a_horizon_color() -> None:
    """Color means horizon everywhere, so any other mark sharing a hex breaks it."""
    others = set(palette.TIER_COLOR.values()) | {
        palette.ACTUAL_INK,
        palette.AXIS_COLOR,
        palette.GRID_COLOR,
        palette.FONT_COLOR,
        palette.PLOT_BGCOLOR,
    }
    assert not others & set(palette.HORIZON_COLOR.values())


def test_the_tier_symbols_avoid_both_horizon_markers() -> None:
    """Circle and diamond mean horizon_1 and horizon_2, so tiers cannot use them."""
    assert not set(palette.TIER_SYMBOL.values()) & set(palette.HORIZON_MARKER.values())


# ================================================
# per-metric rule coverage
# ================================================


@pytest.mark.parametrize(
    "rules",
    [palette.REFERENCE_LINE, palette.SHOW_BENCHMARK, palette.SYMMETRIC_RANGE],
    ids=["reference_line", "show_benchmark", "symmetric_range"],
)
def test_every_displayed_metric_has_a_rule(rules: dict[str, object]) -> None:
    """A displayed metric with no rule raises KeyError at render time, not here."""
    assert not set(palette.METRICS) - set(rules)


def test_every_tier_has_both_a_color_and_a_symbol() -> None:
    """A tier missing either channel draws as a hole in the scatter's legend."""
    assert not set(palette.TIERS) - set(palette.TIER_COLOR)
    assert not set(palette.TIERS) - set(palette.TIER_SYMBOL)
