-- Weather stations are prioritized roughly by proximity to Central Park (40.7829, -73.9654),
-- balanced with data availability. KNYC0 is closest but only has data from 2020+,
-- KTEB0 has full coverage since 1973, KJRB0 since 2016, and 72502 (Newark) since 1973.
--
-- View stations on Google Maps:
-- https://www.google.com/maps/dir/40.7829,-73.9654/40.7789,-73.9692/40.8501,-74.0608/40.7012,-74.0090/40.6833,-74.0000
--
-- Priority order:
--   1. KNYC0 (Yorkville)      - 0.55 km from Central Park
--   2. KTEB0 (Teterboro)      - 10.97 km
--   3. KJRB0 (Wall Street)    - 9.80 km
--   4. 72502 (Newark Airport) - 11.45 km

CREATE OR REPLACE TABLE curated.weather_hourly_fact AS
WITH ranked AS (
    SELECT
        datetime_hour,
        calendar_date,
        temp_celcius,
        relative_humidity_percent,
        precipitation_millimeters,
        snow_depth_centimeters,
        wind_direction_degrees,
        wind_speed_kilometers_per_hour,
        wind_gust_kilometers_per_hour,
        pressure_hectopascals,
        sunshine_minutes,
        cloud_cover_percent,
        weather_condition_code,
        weather_station_id,
        ROW_NUMBER() OVER (
            PARTITION BY datetime_hour
            ORDER BY CASE weather_station_id
                WHEN 'KNYC0' THEN 1
                WHEN 'KTEB0' THEN 2
                WHEN 'KJRB0' THEN 3
                WHEN '72502' THEN 4
                ELSE 5
            END
        ) AS rn
    FROM raw.weather_hourly_fact
    WHERE weather_station_id IN ('KNYC0', 'KTEB0', 'KJRB0', '72502')
)
SELECT
    datetime_hour,
    calendar_date,
    temp_celcius,
    relative_humidity_percent,
    precipitation_millimeters,
    snow_depth_centimeters,
    wind_direction_degrees,
    wind_speed_kilometers_per_hour,
    wind_gust_kilometers_per_hour,
    pressure_hectopascals,
    sunshine_minutes,
    cloud_cover_percent,
    weather_condition_code,
    weather_station_id AS source_station_id,
    current_timestamp AS load_timestamp_utc
FROM ranked
WHERE rn = 1
ORDER BY datetime_hour;
