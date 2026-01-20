CREATE SCHEMA IF NOT EXISTS raw;

CREATE OR REPLACE TABLE raw.weather_daily_fact (
    weather_station_id VARCHAR NOT NULL,
    calendar_date DATE NOT NULL,
    avg_temp_celcius DOUBLE,
    min_temp_celcius DOUBLE,
    max_temp_celcius DOUBLE,
    total_precip_millimeters DOUBLE,
    total_snow_centimeters DOUBLE,
    avg_wind_speed_kilometers_per_hour DOUBLE,
    max_wind_gust_kilometers_per_hour DOUBLE,
    load_timestamp_utc TIMESTAMP,
    PRIMARY KEY (weather_station_id, calendar_date)
);

