SET VARIABLE parquet_files = {{parquet_files}};
SET VARIABLE start_date = {{start_date}};
SET VARIABLE end_date = {{end_date}};

-- STAGING VIEW
-- Note: Meteostat daily returns temp (avg), tmin, tmax, not tavg
-- Note: snwd is snow depth (not total snow), prcp is precip
-- Note: wdir is not available in daily data
CREATE OR REPLACE TEMP VIEW staging_weather_daily AS
SELECT
    station_id AS weather_station_id,
    CAST(time AS DATE) AS calendar_date,
    temp AS avg_temp_celcius,
    tmin AS min_temp_celcius,
    tmax AS max_temp_celcius,
    prcp AS total_precip_millimeters,
    CAST(snwd AS DOUBLE) AS total_snow_centimeters,
    NULL::DOUBLE AS avg_wind_direction_degrees,
    wspd AS avg_wind_speed_kilometers_per_hour,
    wpgt AS max_wind_gust_kilometers_per_hour,
    pres AS avg_pressure_hectopascals,
    CAST(tsun AS DOUBLE) AS sunshine_minutes,
    current_timestamp AS load_timestamp_utc
FROM read_parquet(getvariable('parquet_files'), union_by_name=true)
WHERE (getvariable('start_date') IS NULL OR CAST(time AS DATE) >= CAST(getvariable('start_date') AS DATE))
  AND (getvariable('end_date') IS NULL OR CAST(time AS DATE) <= CAST(getvariable('end_date') AS DATE));

-- MERGE STATEMENT

MERGE INTO raw.weather_daily_fact AS target
USING (
    SELECT
        weather_station_id,
        calendar_date,
        avg_temp_celcius,
        min_temp_celcius,
        max_temp_celcius,
        total_precip_millimeters,
        total_snow_centimeters,
        avg_wind_direction_degrees,
        avg_wind_speed_kilometers_per_hour,
        max_wind_gust_kilometers_per_hour,
        avg_pressure_hectopascals,
        sunshine_minutes,
        load_timestamp_utc
    FROM staging_weather_daily
) AS source
ON target.weather_station_id = source.weather_station_id
    AND target.calendar_date = source.calendar_date
WHEN MATCHED THEN
    UPDATE SET
        avg_temp_celcius = source.avg_temp_celcius,
        min_temp_celcius = source.min_temp_celcius,
        max_temp_celcius = source.max_temp_celcius,
        total_precip_millimeters = source.total_precip_millimeters,
        total_snow_centimeters = source.total_snow_centimeters,
        avg_wind_direction_degrees = source.avg_wind_direction_degrees,
        avg_wind_speed_kilometers_per_hour = source.avg_wind_speed_kilometers_per_hour,
        max_wind_gust_kilometers_per_hour = source.max_wind_gust_kilometers_per_hour,
        avg_pressure_hectopascals = source.avg_pressure_hectopascals,
        sunshine_minutes = source.sunshine_minutes,
        load_timestamp_utc = source.load_timestamp_utc
WHEN NOT MATCHED THEN
    INSERT (
        weather_station_id,
        calendar_date,
        avg_temp_celcius,
        min_temp_celcius,
        max_temp_celcius,
        total_precip_millimeters,
        total_snow_centimeters,
        avg_wind_direction_degrees,
        avg_wind_speed_kilometers_per_hour,
        max_wind_gust_kilometers_per_hour,
        avg_pressure_hectopascals,
        sunshine_minutes,
        load_timestamp_utc
    )
    VALUES (
        source.weather_station_id,
        source.calendar_date,
        source.avg_temp_celcius,
        source.min_temp_celcius,
        source.max_temp_celcius,
        source.total_precip_millimeters,
        source.total_snow_centimeters,
        source.avg_wind_direction_degrees,
        source.avg_wind_speed_kilometers_per_hour,
        source.max_wind_gust_kilometers_per_hour,
        source.avg_pressure_hectopascals,
        source.sunshine_minutes,
        source.load_timestamp_utc
    );
