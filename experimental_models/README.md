# Experimental models

Callables used only for notebook experimentation. The rule that puts one here rather
than in `src/fcstnyctaxi/models/`, stated once and only here: **the rule is about
callables, not models.** A model's pipeline callable lives in `fcstnyctaxi/models/`
and ships in the image; callables serving development work live here. A model still
moves when it enters `model_roles`, and lightgbm holding one callable of each kind is
the normal case for a model under active development, not an exception.

## The two callable contracts

A callable cannot tell the two frames apart, so one module cannot serve both.

| contract | `future_x_df` is | who selects columns |
| --- | --- | --- |
| development, here | the raw `ds`-keyed fiscal calendar | the callable, from its own constant |
| pipeline, `fcstnyctaxi/models/` | a frame keyed `["unique_id", "ds"]`, already trimmed | the impl, from `model_settings.exog_features` |

tsbricks permits the split: `invoke_model` passes two positional arguments and the
keyword `future_x_df`, and never inspects the frame. That keyword is the one thing
both sides share, so renaming it on either side would hide the divergence.

`lightgbm_weekly_dev.py` is what the split cost: the notebooks resolve their callables
by dotted string and must keep working against a raw calendar, so the development
halves were copied here and the configs repointed. `src/fcstnyctaxi/lib/calibration.py`
is written to this contract and has no consumer under `src/`.

## "Experimental" is deployment status, not code quality

Nothing here is a draft. These models produce leaderboard numbers that inform real
decisions, and `autoets_26week` is a registered run that `compare_sidecars()`
checks later runs against. What they are not is *deployed*: they are absent from
`model_roles`, so no Training component ever resolves them, and the Train image
does not carry their dependencies.

A model leaves this directory by being chosen, not by being finished.

## Why the repo root

This directory sits outside `src/fcstnyctaxi/` on purpose.
`[tool.hatch.build.targets.wheel]` packages `src/fcstnyctaxi` and nothing else, so
a model here cannot reach the image by accident, and its dependencies have no
reason to enter the `train` dependency group.

The cost is that `experimental_models.*` resolves only off `sys.path`. That is
what `sys.path.insert(0, project_root)` is for in `notebooks/backtest_weekly.py`,
`notebooks/calibrate_n_estimators.py`, and `notebooks/leaderboard.py`. **That
insert is load-bearing, not vestigial.** Deleting it breaks every dotted path
beginning with `experimental_models`, and `run_config.yaml` selects one today.

`fcstnyctaxi.*` needs no such help: the editable install puts `src/` on `sys.path`
permanently, which is why these modules import `fcstnyctaxi.models._utils` and not
the reverse.
