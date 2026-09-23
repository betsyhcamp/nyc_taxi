# TODO: Once conventions are understood by the team, trim down comments
"""Training's project-owned configuration destinations.

The rules this tree follows are in ``config/README.md``.

Two destinations, one fragment each, per the parity rule: TrainInfraConfig from
``config/train/infra.yaml`` and TrainModelingConfig from
``config/train/modeling.yaml``.

Training's third destination, ``BacktestConfig``, is **tsbricks-owned** and is
not defined here. It composes from ``base/data.yaml`` -> ``train/backtest.yaml``
→ ``train/models/<name>.yaml`` -> runtime overrides, and is the only destination
that layers.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fcstnyctaxi.schemas.run_outputs import (
    ADDITIONAL_EXOG_REQUIRED_COLUMNS,
    CALENDAR_ALLOWED_COLUMNS,
    JOIN_KEYS,
)

DampeningName = Literal["cbrt", "sqrt", "none"]
"""Dampening function names.

The name -> callable map ``DAMPENING_FNS`` lives in ``lib/period_utils.py`` and
imports this, never the reverse: names are contract, callables are
implementation, and the layer rules permit ``lib/`` -> ``schemas/`` while
forbidding the opposite direction.
"""


class ModelRegistry(BaseModel):
    """Prefixes of the Vertex Model's two names; ``infra.yaml`` says why two.

    The pattern is stricter than ``model_id``'s rule: every composed id is legal,
    with no ``--`` seam. Lengths live in ``lib/registry_ids.py``, as no one model
    sees both halves.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    display_name_prefix: str = Field(..., min_length=1)
    model_id_prefix: str = Field(..., pattern=r"^[a-z]([a-z0-9-]*[a-z0-9])?$")


class TrainInfraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    display_name_prefix: str = Field(..., min_length=1)
    model_registry: ModelRegistry


class EvaluationPeriods(BaseModel):
    """Which months the backtest scores.

    ``start_months`` is the one field where ``null`` carries meaning: an
    explicit list pins an experiment, and ``null`` — the default — means derive
    the months from the panel's last complete actual month at composition time.

    ``min_length=1`` makes ``null`` the **only** way to say derive. An empty list
    would mean the same thing, since the consumer spells this
    ``cfg.start_months or derive_start_months(...)`` and ``[]`` is falsy but would
    become "zero months" and fail two layers away on ``forecast_origins``. Forbidding
    the empty list removes the hazard rather than leaving it to be detected later.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_months: list[int] | None = Field(default=None, min_length=1)
    n_start_months: int = Field(..., gt=0)
    start_month_step: int = Field(..., gt=0)
    forecast_horizon_months: int = Field(..., gt=0)


class Tiering(BaseModel):
    """Volume tiers for tier-sliced metrics.

    ``tier_labels`` carries two field-level constraints and no cross-field one:
    the bin count is derived from ``len(tier_labels)`` rather than configured
    separately, because the two could never be usefully independent — too few
    labels makes ``pd.qcut`` raise, and too many silently mislabels via
    ``tier_labels[:effective_tiers]``.
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
        if len(labels) != len(set(labels)):
            raise ValueError(
                f"Tier label list has length {len(labels)} but only"
                f" {len(set(labels))} unique tier labels; duplicates exist."
            )

        return labels


class Weighting(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trailing_weeks: int = Field(..., gt=0)
    dampening: DampeningName


class ModelRoles(BaseModel):
    """Role -> model name.

    A named class rather than ``dict[str, str]``: role keys cannot vary, and
    role order is field declaration order, which is what makes ``model_names``
    a deterministic list — benchmark, then challenger.

    Duplicate *values* are permitted rather than rejected. ``benchmark ==
    challenger`` yields WRMAE = 1.0, which is a legitimate smoke test of the
    whole evaluation path; the name expands once and emits one config.

    Names match ``[a-z0-9_]+`` because they are ``train/models/<name>.yaml``
    file stems now and become KFP task-name components later.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: str = Field(..., min_length=1, pattern=r"^[a-z0-9_]+$")
    challenger: str = Field(..., min_length=1, pattern=r"^[a-z0-9_]+$")


class ModelSettings(BaseModel):
    """One model's input selection and the callables ``final_fit`` resolves.

    Project-owned rather than ``ModelConfig.hyperparameters``, which is
    ``dict[str, Any]`` and so hides a typo from every validation stage.

    ``exog_features`` picks what one model trains on, within what the calendar and
    the additional exogenous contracts let Feature deliver; editing it moves that
    model's numbers. The callables come as a pair: only the model's own save can
    write out the opaque object its fit returns.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    exog_features: list[str] = Field(default_factory=list)
    fit_callable: str | None = None
    save_callable: str | None = None

    @field_validator("exog_features")
    @classmethod
    def _features_must_be_selectable_exogenous_columns(
        cls, features: list[str]
    ) -> list[str]:
        """Reject a join key or a column neither contract declares.

        Keys first: ``unique_id`` is both, and "misspelling" would be the wrong fix.
        """
        keys = sorted(set(features) & set(JOIN_KEYS))
        if keys:
            raise ValueError(
                f"exog_features names join key(s) {keys}, already the frame's key."
            )

        # Less the join keys, so the message cannot contradict the check above by
        # listing as selectable the two names it exists to reject.
        selectable = (
            set(CALENDAR_ALLOWED_COLUMNS) | set(ADDITIONAL_EXOG_REQUIRED_COLUMNS)
        ) - set(JOIN_KEYS)
        unknown = sorted(set(features) - selectable)
        if unknown:
            raise ValueError(
                f"exog_features names {unknown}, in neither the calendar nor the "
                f"additional exogenous contract: {sorted(selectable)}."
            )

        return features

    @model_validator(mode="after")
    def _callables_are_declared_as_a_pair(self) -> "ModelSettings":
        """Reject half a pair, which would fit a model nothing can write out."""
        declared = {
            name
            for name, value in (
                ("fit_callable", self.fit_callable),
                ("save_callable", self.save_callable),
            )
            if value is not None
        }
        if len(declared) == 1:
            missing = {"fit_callable", "save_callable"} - declared
            raise ValueError(
                f"{sorted(declared)[0]} without {sorted(missing)[0]}; "
                "declare both or neither."
            )
        return self


class TrainModelingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_periods: EvaluationPeriods
    tiering: Tiering
    weighting: Weighting
    model_roles: ModelRoles
    model_settings: dict[str, ModelSettings]

    @model_validator(mode="after")
    def _every_role_model_has_settings(self) -> "TrainModelingConfig":
        """Require an entry per role model, since each is backtested. Extras are
        legitimate: a model may keep its entry while it holds no role."""
        missing = sorted(
            set(self.model_roles.model_dump().values()) - set(self.model_settings)
        )
        if missing:
            raise ValueError(
                f"model_settings has no entry for role model(s) {missing}."
            )
        return self

    @model_validator(mode="after")
    def _the_challenger_is_registrable(self) -> "TrainModelingConfig":
        """The challenger is what ``final_fit`` fits, so it declares the pair.

        Keyed on the role: a configurable target would let ``benchmark`` validate.
        """
        challenger = self.model_roles.challenger
        settings = self.model_settings[challenger]
        if settings.fit_callable is None:
            raise ValueError(
                f"challenger {challenger!r} declares no fit_callable and save_callable."
            )
        return self
