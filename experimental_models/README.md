# Experimental models

Models used only for notebook experimentation. The rule that puts a model here
rather than in `src/fcstnyctaxi/models/`, stated once and only here: models the
pipeline runs live in `fcstnyctaxi/models/` and ship in the image, models used
only for experimentation live here, and a model moves when it enters
`model_roles`.

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
