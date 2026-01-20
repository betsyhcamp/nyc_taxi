CREATE OR REPLACE TABLE staging.taxi_yellow_tripdata_fact_silver AS
WITH threshold_cte AS (
    SELECT
        sourcefile_name,
        date_trunc(
            'month',
            CAST(
                regexp_extract(sourcefile_name, '([0-9]{4}-[0-9]{2})') || '-01'
                AS DATE
            ) + INTERVAL 2 MONTH
        ) AS end_date_threshold,
        date_trunc(
            'month',
            CAST(
                regexp_extract(sourcefile_name, '([0-9]{4}-[0-9]{2})') || '-01'
                AS DATE
            ) - INTERVAL 1 MONTH
        ) AS start_date_threshold
    FROM (
        SELECT DISTINCT sourcefile_name
        FROM raw.taxi_yellow_tripdata_fact
    )
)
SELECT raw_taxi.*
FROM raw.taxi_yellow_tripdata_fact AS raw_taxi
INNER JOIN threshold_cte
    ON raw_taxi.sourcefile_name = threshold_cte.sourcefile_name
WHERE
    CAST(tpep_pickup_datetime AS DATE) >= start_date_threshold
    AND CAST(tpep_pickup_datetime AS DATE) <= end_date_threshold;

