import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full", layout_file="layouts/notebook.slides.json")


@app.cell
def _():
    import subprocess
    return (subprocess,)


@app.cell
def _(subprocess):
    #! just run-ddl-queries
    subprocess.call(args=['just', 'run-ddl-queries'])
    #! just download-taxi-data --year-month-start 2025-01 --year-month-end 2025-06
    subprocess.call(
        args=[
            'just',
            'download-taxi-data',
            '--year-month-start',
            '2025-01',
            '--year-month-end',
            '2025-06',
        ]
    )
    #! just merge-taxi-data --year-month-start 2025-01 --year-month-end 2025-06
    subprocess.call(
        args=[
            'just',
            'merge-taxi-data',
            '--year-month-start',
            '2025-01',
            '--year-month-end',
            '2025-06',
        ]
    )
    #! just upsert-dates --start-date 2015-01-01 --end-date 2030-12-31
    subprocess.call(
        args=[
            'just',
            'upsert-dates',
            '--start-date',
            '2015-01-01',
            '--end-date',
            '2030-12-31',
        ]
    )
    #! just build-taxi-curated-layer --year-month-start 2025-01 --year-month-end-inclusive 2025-06
    subprocess.call(
        args=[
            'just',
            'build-taxi-curated-layer',
            '--year-month-start',
            '2025-01',
            '--year-month-end-inclusive',
            '2025-06',
        ]
    )
    return


@app.cell
def _():
    from pathlib import Path

    import duckdb
    from local import load_local_config as load_config

    config = load_config(path=Path("./config.yaml"))
    db_url = f"{config.duckdb_location.as_posix()}"

    print(f"{db_url=}")
    duckdb_cn = duckdb.connect(database=db_url)
    return (duckdb_cn,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(duckdb_cn, mo):
    _df = mo.sql(
        f"""

        """,
        engine=duckdb_cn
    )
    return


@app.cell
def _(duckdb_cn, mo):
    _df = mo.sql(
        f"""
        WITH max_pickup_date_cte AS (
            SELECT
                CAST(
                    date_trunc('month', max_month) + INTERVAL 1 MONTH - INTERVAL 1 DAY
                    AS DATE
                ) AS max_pickup_date
            FROM (
                SELECT
                    MAX(
                        TRY_CAST(
                            NULLIF(
                                regexp_extract(
                                    sourcefile_name,
                                    '([0-9]{4}-[0-9]{2})'
                                ),
                                ''
                            ) || '-01'
                            AS DATE
                        )
                    ) AS max_month
                FROM curated.taxi_yellow_tripdata_fact
            )
        ),
        daily_timeseries_cte AS (
            SELECT
                trip.pickup_taxi_zone_id,
                trip.pickup_date,
                COUNT(trip.trip_id) AS number_ride_pickups
            FROM curated.taxi_yellow_tripdata_fact AS trip
            LEFT JOIN curated.taxi_zone_dim AS zone
                ON trip.pickup_taxi_zone_id = zone.taxi_zone_id
            WHERE
                zone.borough = 'Manhattan'
                AND NOT (trip.fare_amount = 0 OR trip.total_amount = 0)
                AND NOT (trip.trip_distance = 0 AND trip.total_amount <= 0)
                AND pickup_date <= (
                    SELECT max_pickup_date
                    FROM max_pickup_date_cte
                )
            GROUP BY
                pickup_taxi_zone_id,
                pickup_date
        ),
        min_date_bound_cte AS (
            SELECT
                pickup_taxi_zone_id,
                MIN(pickup_date) AS min_pickup_date
            FROM daily_timeseries_cte
            GROUP BY pickup_taxi_zone_id
        ),
        pickup_taxi_zone_id_cal AS (
            SELECT
                mn.pickup_taxi_zone_id,
                d.calendar_date
            FROM min_date_bound_cte AS mn
            CROSS JOIN max_pickup_date_cte AS mx
            INNER JOIN curated.date_dim AS d
                ON
                    mn.min_pickup_date <= d.calendar_date
                    AND mx.max_pickup_date >= d.calendar_date
        )
        SELECT
            p.pickup_taxi_zone_id,
            p.calendar_date AS pickup_date,
            d.calendar_year,
            d.calendar_month,
            d.day_of_week,
            d.day_of_week_name,
            d.is_weekend,
            d.is_holiday,
            d.holiday_name,
            d.is_daylight_savings,
            COALESCE(t.number_ride_pickups, 0) AS number_ride_pickups
        FROM pickup_taxi_zone_id_cal AS p
        LEFT JOIN daily_timeseries_cte AS t
            ON
                p.calendar_date = t.pickup_date
                AND p.pickup_taxi_zone_id = t.pickup_taxi_zone_id
        LEFT JOIN curated.date_dim AS d
            ON p.calendar_date = d.calendar_date
        ORDER BY
            p.pickup_taxi_zone_id,
            p.calendar_date;
        """,
        engine=duckdb_cn
    )
    return


@app.cell
def _(duckdb_cn, mo):
    _df = mo.sql(
        f"""
        SELECT
            CAST(
                date_trunc('month', max_month) + INTERVAL 1 MONTH - INTERVAL 1 DAY
                AS DATE
            ) AS max_pickup_date
        FROM (
            SELECT
                MAX(
                    TRY_CAST(
                        NULLIF(
                            regexp_extract(
                                sourcefile_name,
                                '([0-9]{4}-[0-9]{2})'
                            ),
                            ''
                        ) || '-01'
                        AS DATE
                    )
                ) AS max_month
            FROM curated.taxi_yellow_tripdata_fact
        )
        """,
        engine=duckdb_cn
    )
    return


@app.cell
def _(duckdb_cn, mo):
    _df = mo.sql(
        f"""
        SELECT
                MAX(
                    TRY_CAST(
                        NULLIF(
                            regexp_extract(
                                sourcefile_name,
                                '([0-9]{4}-[0-9]{2})'
                            ),
                            ''
                        ) || '-01'
                        AS DATE
                    )
                ) AS max_month
            FROM curated.taxi_yellow_tripdata_fact
        """,
        engine=duckdb_cn
    )
    return


@app.cell
def _(duckdb_cn, mo):
    _df = mo.sql(
        f"""
        SELECT 
            REGEXP_EXTRACT(sourcefile_name, '([0-9]{4}-[0-9]{2})')  as year_month
        FROM curated.taxi_yellow_tripdata_fact LIMIT 15;
        """,
        engine=duckdb_cn
    )
    return


@app.cell
def _(duckdb_cn):
    duckdb_cn.close()
    return


@app.cell
def _(duckdb_cn, mo):
    _df = mo.sql(
        f"""
        SELECT 
            REGEXP_EXTRACT(sourcefile_name, '([0-9]{4}-[0-9]{2})')  as year_month
        FROM curated.taxi_yellow_tripdata_fact LIMIT 15;
        """,
        engine=duckdb_cn
    )
    return


@app.cell
def _(duckdb_cn, mo):
    _df = mo.sql(
        f"""
        SELECT REGEXP_EXTRACT(sourcefile_name, '([0-9]{4}-[0-9]{2})') AS year_month FROM curated.taxi_yellow_tripdata_fact LIMIT 15;
        """,
        engine=duckdb_cn
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
