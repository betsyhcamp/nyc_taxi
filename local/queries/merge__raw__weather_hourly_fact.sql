SET VARIABLE parquet_files = {{parquet_files}};
SET VARIABLE start_date = {{start_date}};
SET VARIABLE end_date = {{end_date}};

-- STAGING VIEW
CREATE OR REPLACE TEMP VIEW staging_weather_hourly AS
SELECT
    station_id AS weather_station_id,
    time AS datetime_hour,
    CAST(time AS DATE) AS calendar_date,
    temp AS temp_celcius,
    rhum AS relative_humidity_percent,
    prcp AS precipitation_millimeters,
    snwd AS snow_depth_centimeters,
    wdir AS wind_direction_degrees,
    wspd AS wind_speed_kilometers_per_hour,
    wpgt AS wind_gust_kilometers_per_hour,
    pres AS pressure_hectopascals,
    tsun AS sunshine_minutes,
    cldc AS cloud_cover_percent,
    CAST(coco AS INTEGER) AS weather_condition_code,
    current_timestamp AS load_timestamp_utc
FROM read_parquet(getvariable('parquet_files'), union_by_name=true)
WHERE (getvariable('start_date') IS NULL OR CAST(time AS DATE) >= CAST(getvariable('start_date') AS DATE))
  AND (getvariable('end_date') IS NULL OR CAST(time AS DATE) <= CAST(getvariable('end_date') AS DATE));

-- MERGE STATEMENT

MERGE INTO raw.weather_hourly_fact AS target
USING (
    SELECT
        weather_station_id,
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
        load_timestamp_utc
    FROM staging_weather_hourly
) AS source
ON target.weather_station_id = source.weather_station_id
    AND target.datetime_hour = source.datetime_hour
WHEN MATCHED THEN
    UPDATE SET
        calendar_date = source.calendar_date,
        temp_celcius = source.temp_celcius,
        relative_humidity_percent = source.relative_humidity_percent,
        precipitation_millimeters = source.precipitation_millimeters,
        snow_depth_centimeters = source.snow_depth_centimeters,
        wind_direction_degrees = source.wind_direction_degrees,
        wind_speed_kilometers_per_hour = source.wind_speed_kilometers_per_hour,
        wind_gust_kilometers_per_hour = source.wind_gust_kilometers_per_hour,
        pressure_hectopascals = source.pressure_hectopascals,
        sunshine_minutes = source.sunshine_minutes,
        cloud_cover_percent = source.cloud_cover_percent,
        weather_condition_code = source.weather_condition_code,
        load_timestamp_utc = source.load_timestamp_utc
WHEN NOT MATCHED THEN
    INSERT (
        weather_station_id,
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
        load_timestamp_utc
    )
    VALUES (
        source.weather_station_id,
        source.datetime_hour,
        source.calendar_date,
        source.temp_celcius,
        source.relative_humidity_percent,
        source.precipitation_millimeters,
        source.snow_depth_centimeters,
        source.wind_direction_degrees,
        source.wind_speed_kilometers_per_hour,
        source.wind_gust_kilometers_per_hour,
        source.pressure_hectopascals,
        source.sunshine_minutes,
        source.cloud_cover_percent,
        source.weather_condition_code,
        source.load_timestamp_utc
    );
