-- should use revenue rather than ride count but want to set up
-- eval script using dummy data so just added fiscal cal info just to create dummy data


WITH max_pickup_date_cte AS (
    SELECT
        DATE_SUB(
            DATE_ADD(
                MAX(
                    DATE(
                        REGEXP_EXTRACT(sourcefile_name, r'(\d{4}-\d{2})') || '-01'
                    )
                ),
                INTERVAL 1 MONTH
            ),
            INTERVAL 1 DAY
        ) AS max_pickup_date
    FROM `nyc-taxi-ehc.curated.taxi_yellow_tripdata_fact`
),

daily_timeseries_cte AS (
    SELECT
        trip.pickup_taxi_zone_id,
        trip.pickup_date,
        COUNT(trip.trip_id) AS number_ride_pickups
    FROM `nyc-taxi-ehc.curated.taxi_yellow_tripdata_fact` AS trip
    LEFT JOIN `nyc-taxi-ehc.curated.taxi_zone_dim` AS zone
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
    INNER JOIN `nyc-taxi-ehc.curated.date_dim` AS d
        ON
            mn.min_pickup_date <= d.calendar_date
            AND mx.max_pickup_date >= d.calendar_date
),

month_pickup_cte AS (
    SELECT DISTINCT
        dts.pickup_date AS max_pickup_date,
        d.fiscal_year_month
    FROM daily_timeseries_cte AS dts
    INNER JOIN `nyc-taxi-ehc.curated.date_dim` AS d
        ON dts.pickup_date = d.calendar_date
    WHERE dts.pickup_date = (
        SELECT MAX(pickup_date)
        FROM daily_timeseries_cte
    )
),

month_boundary_cte AS (
    SELECT
        fiscal_year_month,
        MAX(calendar_date) AS max_cal_date
    FROM `nyc-taxi-ehc.curated.date_dim`
    WHERE
        fiscal_year_month = (
            SELECT fiscal_year_month
            FROM month_pickup_cte
        )
    GROUP BY fiscal_year_month

),

month_cutoff_cte AS (
    SELECT
        CASE
            WHEN mp.max_pickup_date = mb.max_cal_date THEN mb.fiscal_year_month
            ELSE
                CAST(FORMAT_DATE(
                    '%Y%m', DATE_SUB(
                        DATE(
                            CAST(FLOOR(mp.fiscal_year_month / 100) AS INT64),  -- year
                            CAST(RIGHT(CAST(mp.fiscal_year_month AS STRING), 2) AS INT64),  -- month
                            1
                        ),
                        INTERVAL 1 MONTH
                    )
                )
                AS INT64)
        END AS fiscal_year_month_cutoff
    FROM month_pickup_cte AS mp
    INNER JOIN month_boundary_cte AS mb
        ON mp.fiscal_year_month = mb.fiscal_year_month
)

SELECT
    p.pickup_taxi_zone_id,
    p.calendar_date AS pickup_date,
    d.day_of_fiscal_month,
    d.fiscal_week,
    d.fiscal_week_of_month,
    d.fiscal_month,
    d.fiscal_year,
    d.fiscal_year_month,
    d.day_of_week,
    d.day_of_week_name,
    d.is_weekend,
    d.is_holiday,
    d.holiday_name,
    d.is_daylight_savings,
    DATE_TRUNC(d.calendar_date, WEEK (SUNDAY)) AS fiscal_week_start_date,
    (d.fiscal_year * 100) + d.fiscal_week AS fiscal_year_week,
    COALESCE(t.number_ride_pickups, 0) AS number_ride_pickups
FROM pickup_taxi_zone_id_cal AS p
LEFT JOIN daily_timeseries_cte AS t
    ON
        p.calendar_date = t.pickup_date
        AND p.pickup_taxi_zone_id = t.pickup_taxi_zone_id
LEFT JOIN `nyc-taxi-ehc.curated.date_dim` AS d
    ON p.calendar_date = d.calendar_date
WHERE d.fiscal_year_month <= (SELECT fiscal_year_month_cutoff FROM month_cutoff_cte)
ORDER BY
    p.pickup_taxi_zone_id,
    p.calendar_date
