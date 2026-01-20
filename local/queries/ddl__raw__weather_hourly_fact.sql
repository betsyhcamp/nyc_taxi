CREATE SCHEMA IF NOT EXISTS raw;

CREATE OR REPLACE TABLE raw.weather_hourly_fact (
    weather_station_id VARCHAR NOT NULL,
    datetime_hour TIMESTAMP NOT NULL,
    calendar_date DATE NOT NULL,
    temp_celcius DOUBLE,
    precipitation_millimeters DOUBLE,
    snowfall_centimeters DOUBLE,
    wind_direction_degrees DOUBLE,
    wind_speed_kilometers_per_hour DOUBLE,
    wind_gust_kilometers_per_hour DOUBLE,
    load_timestamp_utc TIMESTAMP,
    PRIMARY KEY (weather_station_id, datetime_hour)
);

