"""Does backtest_impl reproduce the notebook?

A copy of backtest_weekly.py's compare_sidecars primitives; consolidate when the
regression gate needs them. Diverges: seven files not five, refuses an impl prefix
with no completion marker, _dict_diff returns paths, exits nonzero. `new` is the
impl, `ref` the notebook.
"""

import argparse

import fsspec
import pandas as pd
import yaml

# The notebook injects the calendar URI; the production tree passes calendar_df
# directly, so it declares calendar_source null.
_EXPECTED_CONFIG_DIFFS = ("aggregation.calendar_source",)

# Redeclared: backtest_impl's copy is private and the runner already spells it out.
_MANIFEST_FILENAME = "backtest_manifest.json"
_CONFIG_FILENAME = "composed_config.yaml"

_COMPARED_FRAMES = (
    "time_series_snapshot.parquet",
    "fiscal_calendar.parquet",
    "monthly_series.parquet",
    "raw_cv_forecasts.parquet",
    "metrics.parquet",
    "monthly_forecast_components.parquet",
)

# Reported, not omitted: a one-sided file must not read as a missing output.
_ONE_SIDED_FILES = {
    "run_metadata.json": "notebook",
    _MANIFEST_FILENAME: "impl",
}


def _read(uri: str, name: str) -> pd.DataFrame | None:
    """Read a sidecar parquet, or None if absent, so one missing file fails its
    own check rather than aborting the report.
    """
    try:
        return pd.read_parquet(f"{uri}{name}")
    except Exception:
        return None


def _decategorize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten categories to object: a category sorts by its category order, so
    identical values compare unequal when the orders differ. Dtypes are compared
    separately, so this hides nothing real.
    """
    cat_cols = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
    return df.astype({c: "object" for c in cat_cols}) if cat_cols else df


def _compare_frames(new_df: pd.DataFrame, ref_df: pd.DataFrame) -> tuple[bool, str]:
    """Sort both frames by every column, then compare column set, dtypes, values.

    Sorting by every column is deterministic without naming key columns, so it
    survives an upstream schema change.

    Returns:
        (passed, detail) naming what differed: columns, dtypes, or max abs delta.
    """
    new_cols, ref_cols = set(new_df.columns), set(ref_df.columns)
    if new_cols != ref_cols:
        return False, (
            f"column sets differ: only_new={sorted(new_cols - ref_cols)}, "
            f"only_ref={sorted(ref_cols - new_cols)}"
        )

    cols = sorted(new_cols)
    problems = []

    dtype_diffs = {
        c: f"{new_df[c].dtype} vs {ref_df[c].dtype}"
        for c in cols
        if new_df[c].dtype != ref_df[c].dtype
    }
    if dtype_diffs:
        problems.append(f"dtype mismatches: {dtype_diffs}")

    if len(new_df) != len(ref_df):
        problems.append(f"row counts differ: new={len(new_df)} ref={len(ref_df)}")
        return False, "; ".join(problems)

    a = _decategorize(new_df[cols]).sort_values(cols).reset_index(drop=True)
    b = _decategorize(ref_df[cols]).sort_values(cols).reset_index(drop=True)

    value_diffs = []
    for c in cols:
        # Not Series.equals: dtype-strict, so an int-to-float column would also
        # report "values differ (0 rows differ)". Not a bare ==: NaN != NaN.
        if ((a[c] == b[c]) | (a[c].isna() & b[c].isna())).all():
            continue
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            value_diffs.append(f"{c} (max abs delta {(a[c] - b[c]).abs().max():.6g})")
        else:
            value_diffs.append(f"{c} ({int((a[c] != b[c]).sum())} rows differ)")
    if value_diffs:
        problems.append("values differ in " + ", ".join(value_diffs))

    return (not problems), ("identical" if not problems else "; ".join(problems))


def _dict_diff(new_obj, ref_obj, path: str = "") -> list[tuple[str, str]]:
    """Leaf-level diff of two parsed-YAML structures, as (path, detail) pairs.

    Parsed rather than textual: YAML formatting is not a config change. The path
    stays a value so the caller can match it without parsing prose apart.
    """
    if isinstance(new_obj, dict) and isinstance(ref_obj, dict):
        out: list[tuple[str, str]] = []
        for key in sorted(set(new_obj) | set(ref_obj), key=str):
            sub = f"{path}.{key}" if path else str(key)
            if key not in new_obj:
                out.append((sub, "missing in new"))
            elif key not in ref_obj:
                out.append((sub, "missing in ref"))
            else:
                out += _dict_diff(new_obj[key], ref_obj[key], sub)
        return out
    if new_obj == ref_obj:
        return []
    return [(path, f"new={new_obj!r} ref={ref_obj!r}")]


def _require_prefix(uri: str, flag: str) -> None:
    """Reject a prefix with no trailing slash rather than normalizing it.

    The slash marks a prefix, not an object name; normalizing hides the error and
    the marker check would blame the sidecar for a typo in the flag.
    """
    if not uri.endswith("/"):
        raise ValueError(f"{flag} {uri!r} must end in '/', which marks a prefix.")


def _require_completed_impl_sidecar(impl_prefix: str) -> None:
    """Refuse an impl prefix whose completion marker is absent.

    Nothing clears the sidecar directory and the impl deletes its own marker before
    writing, so absence means the files may come from two attempts. Skipping the
    marker as one-sided is about comparing it; this is about trusting the directory.
    """
    fs, path = fsspec.url_to_fs(f"{impl_prefix}{_MANIFEST_FILENAME}")
    if not fs.exists(path):
        raise ValueError(
            f"No {_MANIFEST_FILENAME} under {impl_prefix}: the impl writes it last "
            "and clears it on rerun, so this prefix never completed and its files "
            "may come from two different attempts."
        )


def _compare_configs(impl_prefix: str, notebook_prefix: str) -> tuple[bool, str]:
    """Diff the two composed configs, separating expected differences from the rest.

    An absent expected difference is reported, not failed: it says nothing about
    whether the numbers reproduce.

    Returns:
        (passed, detail) where passed means no unexpected difference.
    """
    try:
        with fsspec.open(f"{impl_prefix}{_CONFIG_FILENAME}", "r") as handle:
            new_cfg = yaml.safe_load(handle)
        with fsspec.open(f"{notebook_prefix}{_CONFIG_FILENAME}", "r") as handle:
            ref_cfg = yaml.safe_load(handle)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    diffs = _dict_diff(new_cfg, ref_cfg)
    expected = [path for path, _ in diffs if path in _EXPECTED_CONFIG_DIFFS]
    unexpected = [
        f"{path}: {detail}"
        for path, detail in diffs
        if path not in _EXPECTED_CONFIG_DIFFS
    ]
    seen = f"{len(expected)} of {len(_EXPECTED_CONFIG_DIFFS)} expected diffs present"
    if unexpected:
        return False, (
            f"{seen}; {len(unexpected)} unexpected: " + "; ".join(unexpected[:5])
        )
    return True, seen


def _parse_args() -> argparse.Namespace:
    """The two sidecar prefixes, local or gs://.

    No --model: each prefix already names one model's sidecar.
    """
    parser = argparse.ArgumentParser(
        description="Compare one model's impl sidecar against the notebook's."
    )
    parser.add_argument(
        "--impl-prefix",
        required=True,
        help="Sidecar prefix backtest_impl wrote, ending in '/'.",
    )
    parser.add_argument(
        "--notebook-prefix",
        required=True,
        help="Sidecar prefix backtest_weekly.py wrote, ending in '/'.",
    )
    return parser.parse_args()


def main() -> None:
    """Report whether the two sidecars agree, and exit nonzero when they do not.

    Every check runs and reports, so one invocation surfaces every difference.

    Raises:
        ValueError: If a prefix lacks its trailing slash, or the impl prefix
            carries no completion marker.
        SystemExit: With status 1 when any check failed.
    """
    args = _parse_args()
    _require_prefix(args.impl_prefix, "--impl-prefix")
    _require_prefix(args.notebook_prefix, "--notebook-prefix")
    _require_completed_impl_sidecar(args.impl_prefix)

    rows: list[dict[str, str]] = []

    def record(check: str, passed: bool, detail: str) -> None:
        rows.append(
            {"check": check, "status": "PASS" if passed else "FAIL", "detail": detail}
        )

    # First: a config difference explains the frame differences below.
    record(_CONFIG_FILENAME, *_compare_configs(args.impl_prefix, args.notebook_prefix))

    for filename in _COMPARED_FRAMES:
        new_df = _read(args.impl_prefix, filename)
        ref_df = _read(args.notebook_prefix, filename)
        if new_df is None or ref_df is None:
            absent = "impl" if new_df is None else "notebook"
            record(filename, False, f"unreadable in the {absent} sidecar")
        else:
            record(filename, *_compare_frames(new_df, ref_df))

    # SKIP rather than PASS: nothing was checked.
    for filename, side in _ONE_SIDED_FILES.items():
        rows.append(
            {
                "check": filename,
                "status": "SKIP",
                "detail": f"{side} writes it; the other side has no counterpart",
            }
        )

    report = pd.DataFrame(rows, columns=["check", "status", "detail"])
    print(report.to_string(index=False))

    equivalent = not bool((report["status"] == "FAIL").any())
    print(f"\nEQUIVALENT: {equivalent}")
    if not equivalent:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
