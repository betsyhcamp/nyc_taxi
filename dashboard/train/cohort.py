import pandas as pd

from fcstnyctaxi.lib.origin_modeling_table.builders import enumerate_origins

COHORT_KEYS = ["weeks_in_month", "weeks_actualized"]


def attach_cohorts(fold_metrics: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Each fold row's position cohort: the weeks its origin's horizon_1 target
    month has, and how many of them were actualized at the origin.

    Joined on the origin alone, not the fold key. A horizon_2 row predicts the
    month after its origin's spine entry, so a fold-key join would leave it
    without a cohort; on the origin, both horizons share one, as the strip needs.
    """
    spine = enumerate_origins(calendar)[["forecast_origin_date", *COHORT_KEYS]]
    return fold_metrics.merge(spine, on="forecast_origin_date", how="left")
