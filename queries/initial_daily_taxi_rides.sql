WITH max_pickup_date_cte AS (
  SELECT DATE_SUB(
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
    pickup_taxi_zone_id,
    pickup_date,
    COUNT(trip_id) AS number_ride_pickups
  FROM `nyc-taxi-ehc.curated.taxi_yellow_tripdata_fact` AS trip
  LEFT JOIN `nyc-taxi-ehc.curated.taxi_zone_dim` AS zone
  ON trip.pickup_taxi_zone_id = zone.taxi_zone_id
  WHERE zone.borough='Manhattan'
    AND (trip.fare_amount !=0 OR trip.total_amount!=0)
    AND (trip.trip_distance != 0 AND trip.total_amount <= 0)
    AND pickup_date <= (
      SELECT max_pickup_date
      FROM max_pickup_date_cte
    )
  GROUP BY 
    pickup_taxi_zone_id, 
    pickup_date
)
SELECT
  t.pickup_taxi_zone_id,
  t.pickup_date,
  t.number_ride_pickups,
  d.calendar_year,
  d.calendar_month,
  d.day_of_week,
  d.day_of_week_name,
  d.is_weekend,
  d.is_holiday,
  d.holiday_name,
  d.is_daylight_savings
FROM daily_timeseries_cte AS t
LEFT JOIN `nyc-taxi-ehc.curated.date_dim` AS d
ON t.pickup_date = d.calendar_date