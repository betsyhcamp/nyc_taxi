WITH max_pickup_date_cte AS (
    SELECT
        CAST(
            MAX(
                CAST(
                    regexp_extract(sourcefile_name, '([0-9]{4}-[0-9]{2})') || '-01'
                    AS DATE
                )
            ) + INTERVAL 1 MONTH - INTERVAL 1 DAY
            AS DATE
        ) AS max_pickup_date
    FROM curated.taxi_yellow_tripdata_fact
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

