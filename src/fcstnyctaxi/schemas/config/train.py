"""Training's project-owned configuration destinations.

Two destinations, one fragment each, per configs_schemas_images §4.6's parity
rule: TrainInfraConfig from ``config/train/infra.yaml`` and TrainModelingConfig
from ``config/train/modeling.yaml``.

Training's third destination, ``BacktestConfig``, is **tsbricks-owned** and is
not defined here. It composes from ``base/data.yaml`` → ``train/backtest.yaml``
→ ``train/models/<name>.yaml`` → runtime overrides, and is the only destination
that layers.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DampeningName = Literal["cbrt", "sqrt", "none"]
"""Dampening function names.

The name → callable map ``DAMPENING_FNS`` lives in ``lib/period_utils.py`` and
imports this, never the reverse: names are contract, callables are
implementation, and pipeline_architecture §3 permits ``lib/`` → ``schemas/``
while forbidding the opposite direction.
"""


class FeatureSource(BaseModel):
    """Filenames only — the directory is derived.

    ``<env>/feature/<feature_run_id>/data_prep/`` is built by ``lib/io.py``'s
    ``build_feature_uri`` from the env selector and the ``feature_run_id``
    parameter, so a ``root`` field would be a second source of a derived fact
    (configs_schemas_images §4.3).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    panel_filename: str = Field(..., min_length=1)
    calendar_filename: str = Field(..., min_length=1)
    pointer_filename: str = Field(..., min_length=1)


class ModelRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    display_name: str = Field(..., min_length=1)


class TrainInfraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    display_name_prefix: str = Field(..., min_length=1)
    feature_source: FeatureSource
    model_registry: ModelRegistry


class EvaluationPeriods(BaseModel):
    """Which months the backtest scores.

    ``start_months`` is the one field where ``null`` carries meaning: an
    explicit list pins an experiment, and ``null`` — the default — means derive
    the months from the panel's last complete actual month at composition time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_months: list[int] | None = None
    n_start_months: int = Field(..., gt=0)
    start_month_step: int = Field(..., gt=0)
    forecast_horizon_months: int = Field(..., gt=0)


class Tiering(BaseModel):
    """Volume tiers for tier-sliced metrics.

    ``tier_labels`` carries two field-level constraints and no cross-field one:
    the bin count is derived from ``len(tier_labels)`` rather than configured
    separately, because the two could never be usefully independent (spec
    §4.22).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trailing_weeks: int = Field(..., gt=0)
    tier_labels: list[str] = Field(..., min_length=2)

    @field_validator("tier_labels")
    @classmethod
    def _labels_must_be_unique(cls, labels: list[str]) -> list[str]:
        """Reject a tier vocabulary that repeats a label.

        Duplicates merge two bins in every downstream groupby while looking
        correct, so this cannot be left to inspection.
        """
        # TODO(human)
        return labels


class Weighting(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trailing_weeks: int = Field(..., gt=0)
    dampening: DampeningName


class ModelRoles(BaseModel):
    """Role → model name.

    A named class rather than ``dict[str, str]``: role keys cannot vary, and
    role order is field declaration order, which is what makes ``model_names``
    a deterministic list — benchmark, then challenger (spec §4.15).

    Duplicate *values* are permitted rather than rejected. ``benchmark ==
    challenger`` yields WRMAE = 1.0, which is a legitimate smoke test of the
    whole evaluation path; the name expands once and emits one config.

    Names match ``[a-z0-9_]+`` because they are ``train/models/<name>.yaml``
    file stems now and become KFP task-name components in PR 4.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: str = Field(..., min_length=1, pattern=r"^[a-z0-9_]+$")
    challenger: str = Field(..., min_length=1, pattern=r"^[a-z0-9_]+$")


class TrainModelingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_periods: EvaluationPeriods
    tiering: Tiering
    weighting: Weighting
    model_roles: ModelRoles
