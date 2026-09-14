"""
Composing and recording are one step because they share a panel read to derive origins.
Derived origins injected as runtime override. Artifacts & lineage in `run_identity.json`
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pandas as pd

from fcstnyctaxi.lib.column_checks import require_columns
from fcstnyctaxi.lib.config.bindings import (
    environment_bindings,
    model_names_from_roles,
    require_known_environment,
    train_backtest_bindings,
    train_infra_bindings,
    train_modeling_bindings,
)
from fcstnyctaxi.lib.config.composition import (
    RUNTIME_SOURCE,
    ComposedConfig,
    compose_config,
    save_config,
)
from fcstnyctaxi.lib.period_utils import (
    derive_start_months,
    generate_origins_for_periods,
    last_complete_actual_month,
)
from fcstnyctaxi.schemas.config.train import EvaluationPeriods, TrainModelingConfig
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity

# Only column required here. Moves to schemas/run_identity.py, if a 2nd module names it
_LINEAGE_COLUMN = "feature_run_id"


@dataclass(frozen=True)
class SourcedPath:
    """A filepath paired w/ the URI it represents. Can't check both are same object."""

    path: Path
    uri: str


@dataclass(frozen=True)
class ComposeConfigsSummary:
    """Records what is composed for observability, never correctness.

    Only the compose_configs_impl opens the files this describes. A field belongs here
    only if the caller cannot recompute it AND would look at it — hence no lineage id.
    """

    n_origins: int
    first_origin: str
    last_origin: str
    last_complete_actual_month: int
    start_months: list[int]
    model_names: list[str]

    def as_dict(self) -> dict[str, Any]:
        """Coerce to JSON-safe primitives for KFP's `artifact.metadata`.

        Hand written so JSON-safety is guaranteed continuing to coerce the two integer
        fields off the calendar's `int64` & every value JSON-serialisable.
        """
        return {
            "n_origins": self.n_origins,
            "first_origin": self.first_origin,
            "last_origin": self.last_origin,
            "last_complete_actual_month": int(self.last_complete_actual_month),
            "start_months": [int(month) for month in self.start_months],
            "model_names": list(self.model_names),
        }


def compose_train_static_configs(
    config_dir: Path, env: str
) -> tuple[ComposedConfig, ComposedConfig, ComposedConfig]:
    """Compose the destinations that need no data, in the order returned.

    `ComposedConfig` rather than bare models, because the manifest needs
    `value_sources` and `config_files`; a caller wanting the instance takes
    `.config`. A plain tuple is safe here since the three are distinct types.

    Args:
        config_dir (Path): Root the bindings' relative paths resolve against.
        env (str): Environment selector, guarded here so the impl, the local
            runner, and `task verify-image` share one message.

    Raises:
        ValueError: If `env` has no `environments/<env>.yaml`, or on any
            composition failure.

    Returns:
        tuple[ComposedConfig, ComposedConfig, ComposedConfig]: Environment,
            infra, and modeling, in that order.
    """
    require_known_environment(config_dir, env)

    return (
        compose_config(config_dir, environment_bindings(env)),
        compose_config(config_dir, train_infra_bindings()),
        compose_config(config_dir, train_modeling_bindings()),
    )


def _single_feature_run_id(frame: pd.DataFrame, frame_name: str) -> str:
    """The one non-null `feature_run_id` a frame carries.

    Nulls are rejected separately because `nunique()` skips them, and
    `require_columns` names which of the two frames lacks the column.
    """
    require_columns(frame, [_LINEAGE_COLUMN], frame_name)
    column = frame[_LINEAGE_COLUMN]

    null_count = int(column.isna().sum())
    if null_count:
        raise ValueError(
            f"{frame_name} has {null_count} row(s) with a null "
            f"{_LINEAGE_COLUMN}, so those rows carry no provenance."
        )

    distinct = sorted(column.unique())
    if len(distinct) != 1:
        raise ValueError(
            f"{frame_name} mixes Feature runs: {len(distinct)} distinct "
            f"{_LINEAGE_COLUMN} values {distinct[:5]}."
        )

    return str(distinct[0])


def _observe_feature_run_id(
    panel_df: pd.DataFrame, calendar_df: pd.DataFrame, expected: str
) -> str:
    """The id both frames carry, checked against the one the caller declared.

    Consistency between frames is checked first, since a shared value must exist
    before it can be compared. `expected` is a claim; the observation gets stamped.
    """
    panel_id = _single_feature_run_id(panel_df, "panel")
    calendar_id = _single_feature_run_id(calendar_df, "calendar")

    if panel_id != calendar_id:
        raise ValueError(
            f"panel feature_run_id {panel_id!r} != calendar {calendar_id!r}; "
            f"origins and actuals would come from different Feature runs."
        )
    if panel_id != expected:
        raise ValueError(
            f"Frames carry feature_run_id {panel_id!r}, not the declared "
            f"{expected!r}: wrong URIs, or wrong bytes at the requested location."
        )

    return panel_id


def _derive_origins(
    panel_df: pd.DataFrame, calendar_df: pd.DataFrame, periods: EvaluationPeriods
) -> tuple[list[dict], list[int], int]:
    """The one runtime override, with the summary fields derived alongside it.

    `start_months` is configured or derived, never both; `last_complete` is passed
    down because the calendar outruns the actuals and cannot say where they stop.
    """
    last_complete = last_complete_actual_month(
        max_actual_date=panel_df["ds"].max(), calendar_df=calendar_df
    )
    start_months = periods.start_months or derive_start_months(
        last_complete_actual_month=last_complete,
        n_start_months=periods.n_start_months,
        start_month_step=periods.start_month_step,
        forecast_horizon_months=periods.forecast_horizon_months,
        calendar_df=calendar_df,
    )
    origins = generate_origins_for_periods(
        start_months=start_months,
        forecast_horizon_months=periods.forecast_horizon_months,
        calendar_df=calendar_df,
        last_complete_actual_month=last_complete,
    )

    return origins, start_months, last_complete


def _build_manifest(
    config_dir: Path, env: str, destinations: dict[str, ComposedConfig]
) -> dict[str, Any]:
    """How this run's configs were assembled — a record nothing reads back.

    Order decides the winner, so each destination keeps its own list while the
    top-level map dedupes; `value_sources` keeps shadowed entries, not just winners.
    """
    return {
        "env": env,
        "config_files": {
            source: f"sha256:{sha256((config_dir / source).read_bytes()).hexdigest()}"
            for composed in destinations.values()
            for source in composed.config_files
            if source != RUNTIME_SOURCE
        },
        "destinations": {
            key: {
                "config_files": composed.config_files,
                "value_sources": composed.value_sources,
            }
            for key, composed in destinations.items()
        },
    }


def compose_configs_impl(
    *,
    config_dir: Path,
    env: str,
    panel: SourcedPath,
    calendar: SourcedPath,
    expected_feature_run_id: str,
    train_run_id: str,
    git_hash: str,
    out_dir: Path,
) -> ComposeConfigsSummary:
    """Compose and emit every Training destination for one run.

    Keyword-only to address possible transposition inherent in large num inputs.
    `expected_feature_run_id` is a claim, checked and then discarded; the
    observed frame value is what `TrainRunIdentity` stamps. Every failure below
    raises before the first write, so a failed run leaves no partial output.

    Args:
        config_dir (Path): Root of the config tree.
        env (str): Environment selector.
        panel (SourcedPath): The actuals; opened for origins, stamped for lineage.
        calendar (SourcedPath): The fiscal calendar, same pairing.
        expected_feature_run_id (str): The caller-declared upstream Feature run,
            compared against both frames and then discarded.
        train_run_id (str): This run's own identifier.
        git_hash (str): The commit that produced this run.
        out_dir (Path): Step directory the emitted configs are written into,
            created if missing. Must sit under the run root, since
            `run_identity.json` is written to its parent.

    Raises:
        ValueError: If `panel` and `calendar` name one file; if `out_dir` is not a
            step directory under `train_run_id`; on a failed lineage check; or on
            any composition failure.
        ValidationError: If an identity field is malformed.

    Returns:
        ComposeConfigsSummary: What this run composed, for a UI node or a log.
    """
    # Simple check for input arg miswiring causing duplication downstream
    if panel.path == calendar.path:
        raise ValueError(
            f"Miswiring error: panel and calendar are the same filepath {panel.path};"
        )
    environment, infra, modeling = compose_train_static_configs(config_dir, env)
    panel_df = pd.read_parquet(panel.path)
    calendar_df = pd.read_parquet(calendar.path)
    feature_run_id = _observe_feature_run_id(
        panel_df, calendar_df, expected_feature_run_id
    )
    modeling_config = cast(TrainModelingConfig, modeling.config)
    origins, start_months, last_complete = _derive_origins(
        panel_df, calendar_df, modeling_config.evaluation_periods
    )
    model_names = model_names_from_roles(modeling_config.model_roles)

    # Two maps over the same objects: filename for the writes, destination key
    # for the manifest. Binding and manifest agree by construction due to read.
    files = {
        "environment.yaml": environment,
        "infra.yaml": infra,
        "modeling.yaml": modeling,
    }
    destinations = {
        environment_bindings(env)[0].destination_key: environment,
        train_infra_bindings()[0].destination_key: infra,
        train_modeling_bindings()[0].destination_key: modeling,
    }
    for model_name in model_names:
        bindings = train_backtest_bindings(model_name)
        composed = compose_config(
            config_dir, bindings, {"cross_validation": {"forecast_origins": origins}}
        )
        files[f"composed_config_{model_name}.yaml"] = composed
        destinations[bindings[0].destination_key] = composed

    # Both built before the first write; on Vertex out_dir is the durable destination.
    manifest = _build_manifest(config_dir, env, destinations)
    identity = TrainRunIdentity(
        git_hash=git_hash,
        feature_run_id=feature_run_id,
        train_run_id=train_run_id,
        panel_uri=panel.uri,
        calendar_uri=calendar.uri,
    )

    # run_identity.json describes the run, not this step. Derived, not passed:
    # a second parameter could disagree with out_dir about the same directory.
    run_dir = out_dir.parent
    if run_dir.name != train_run_id:
        raise ValueError(
            f"out_dir {out_dir} must be a step directory under the run root "
            f"{train_run_id!r}, since run_identity.json is written beside it."
        )

    # All four dumped with save_config's model-branch flags; exclude_none drops nulls.
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, composed in files.items():
        save_config(
            composed.config.model_dump(by_alias=True, exclude_none=True),
            out_dir / filename,
        )

    (run_dir / "run_identity.json").write_text(
        identity.model_dump_json(indent=2) + "\n"
    )
    # `manifest.json` written is the step's completion marker. Keep this write last.
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return ComposeConfigsSummary(
        n_origins=len(origins),
        first_origin=origins[0]["origin"],
        last_origin=origins[-1]["origin"],
        last_complete_actual_month=last_complete,
        start_months=list(start_months),
        model_names=list(model_names),
    )
