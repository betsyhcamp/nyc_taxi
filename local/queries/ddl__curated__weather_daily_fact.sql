-- Curated daily weather fact table with one row per day, coalesced from multiple
-- stations by priority. Provides aggregated metrics for modeling lower-frequency effects.

CREATE SCHEMA IF NOT EXISTS curated;

CREATE TABLE IF NOT EXISTS curated.weather_daily_fact (
    -- Calendar date for the daily aggregated weather metrics; primary key.
    calendar_date DATE NOT NULL,
    -- Average air temperature in degrees Celsius over the full day.
    avg_temp_celcius DOUBLE,
    -- Minimum hourly air temperature in degrees Celsius observed during the day.
    min_temp_celcius DOUBLE,
    -- Maximum hourly air temperature in degrees Celsius observed during the day.
    max_temp_celcius DOUBLE,
    -- Total precipitation in millimeters accumulated during the day.
    total_precip_millimeters DOUBLE,
    -- Total snowfall in centimeters accumulated during the day.
    total_snow_centimeters DOUBLE,
    -- Average wind direction in degrees during the day.
    avg_wind_direction_degrees DOUBLE,
    -- Average wind speed in kilometers per hour across all hourly observations.
    avg_wind_speed_kilometers_per_hour DOUBLE,
    -- Maximum wind gust in kilometers per hour observed during the day.
    max_wind_gust_kilometers_per_hour DOUBLE,
    -- Average atmospheric pressure in hectopascals during the day.
    avg_pressure_hectopascals DOUBLE,
    -- Total sunshine duration in minutes during the day.
    sunshine_minutes DOUBLE,
    -- Station ID that provided this row's data (for lineage/debugging).
    source_station_id VARCHAR NOT NULL,
    -- Timestamp in UTC when row was loaded.
    load_timestamp_utc TIMESTAMP,
    PRIMARY KEY (calendar_date)
);
