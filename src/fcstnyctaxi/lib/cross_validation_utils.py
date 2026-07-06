"mirrors tsbricks, lives here until validated."

import pandas as pd


def sorted_origin_horizon_pairs(origin_horizon_pairs: list[tuple], freq: int | str):
    """Return origin_horizon_pairs sorted ascending by origin.

    Mirrors the defensive sort in tsbricks.backtesting.engine.run_backtest(),
    extracted as a named function so manual backtest loops
    get the same correctness guarantee. Origins are cast to their canonical type
    before sorting, matching run_backtest()'s own normalization.

    Args:
        origin_horizon_pairs: Raw (origin, horizon) pairs from
            cfg.cross_validation.origin_horizon_pairs().
        freq: Forecast frequency. When freq == 1, origins are cast to int;
            otherwise cast to pd.Timestamp.

    Returns:
        The same pairs sorted ascending by origin, with origins normalized to
        int (freq == 1) or pd.Timestamp (otherwise).
    """

    if freq == 1:
        return sorted((int(o), h) for o, h in origin_horizon_pairs)
    else:
        return sorted((pd.Timestamp(o), h) for o, h in origin_horizon_pairs)
