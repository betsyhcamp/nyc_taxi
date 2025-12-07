CREATE OR REPLACE VIEW `nyc-taxi-ehc.staging.taxi_yellow_tripdata_fact_silver` AS (
WITH threshold_cte AS (
  SELECT
  sourcefile_name,
  DATE_TRUNC(
    DATE_ADD(
      DATE(REGEXP_EXTRACT(sourcefile_name, r'(\d{4}-\d{2})') || '-01'),
      INTERVAL 2 MONTH
    ),
    MONTH
  ) AS end_date_threshold,
  DATE_TRUNC(
    DATE_SUB(
      DATE(REGEXP_EXTRACT(sourcefile_name, r'(\d{4}-\d{2})') || '-01'),
      INTERVAL 1 MONTH
    ),
    MONTH
  ) AS start_date_threshold
  FROM (
    SELECT DISTINCT sourcefile_name
    FROM `nyc-taxi-ehc.raw.taxi_yellow_tripdata_fact`
  )
)
  SELECT 
    raw_taxi.*
  FROM `nyc-taxi-ehc.raw.taxi_yellow_tripdata_fact` AS raw_taxi
  JOIN threshold_cte 
  ON raw_taxi.sourcefile_name = threshold_cte.sourcefile_name
  WHERE DATE(tpep_pickup_datetime) >= start_date_threshold
  AND DATE(tpep_pickup_datetime) <= end_date_threshold
)