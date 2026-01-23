-- Curated hourly weather fact table with one row per hour, coalesced from multiple
-- stations by priority. Used to join meteorological features to taxi demand facts.

CREATE SCHEMA IF NOT EXISTS curated;

CREATE TABLE IF NOT EXISTS curated.weather_hourly_fact (
    -- Hour start timestamp in local time; primary key.
    datetime_hour TIMESTAMP NOT NULL,
    -- Calendar date corresponding to datetime_hour; FK to date_dim.calendar_date.
    calendar_date DATE NOT NULL,
    -- Air temperature in degrees Celsius.
    temp_celcius DOUBLE,
    -- Relative humidity as a percentage (0-100).
    relative_humidity_percent DOUBLE,
    -- Total liquid precipitation in millimeters recorded during the hour.
    precipitation_millimeters DOUBLE,
    -- Snow depth in centimeters at time of observation.
    snow_depth_centimeters DOUBLE,
    -- Wind direction in degrees where 0 means north and values increase clockwise.
    wind_direction_degrees DOUBLE,
    -- Sustained wind speed in kilometers per hour.
    wind_speed_kilometers_per_hour DOUBLE,
    -- Maximum wind gust speed in kilometers per hour during the hour.
    wind_gust_kilometers_per_hour DOUBLE,
    -- Atmospheric pressure in hectopascals (hPa).
    pressure_hectopascals DOUBLE,
    -- Sunshine duration in minutes during the hour.
    sunshine_minutes DOUBLE,
    -- Cloud cover as a percentage (0-100).
    cloud_cover_percent DOUBLE,
    -- Weather condition code from Meteostat.
    weather_condition_code INTEGER,
    -- Station ID that provided this row's data (for lineage/debugging).
    source_station_id VARCHAR NOT NULL,
    -- Timestamp in UTC when row was loaded.
    load_timestamp_utc TIMESTAMP,
    PRIMARY KEY (datetime_hour)
);
