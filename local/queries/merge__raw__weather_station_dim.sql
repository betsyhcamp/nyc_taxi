SET VARIABLE weather_station_csv = {{weather_station_csv}};

-- STAGING VIEW
CREATE OR REPLACE TEMP VIEW staging_weather_station AS
SELECT
    weather_station_id,
    name,
    country,
    region,
    latitude,
    longitude,
    elevation_meters,
    timezone,
    distance_from_central_park_km,
    priority_rank,
    current_timestamp AS load_timestamp_utc
FROM read_csv_auto(getvariable('weather_station_csv'), header=true);

-- MERGE STATEMENT

MERGE INTO raw.weather_station_dim AS target
USING (
    SELECT
        weather_station_id,
        name,
        country,
        region,
        latitude,
        longitude,
        elevation_meters,
        timezone,
        distance_from_central_park_km,
        priority_rank,
        load_timestamp_utc
    FROM staging_weather_station
) AS source
ON target.weather_station_id = source.weather_station_id
WHEN MATCHED THEN
    UPDATE SET
        name = source.name,
        country = source.country,
        region = source.region,
        latitude = source.latitude,
        longitude = source.longitude,
        elevation_meters = source.elevation_meters,
        timezone = source.timezone,
        distance_from_central_park_km = source.distance_from_central_park_km,
        priority_rank = source.priority_rank,
        load_timestamp_utc = source.load_timestamp_utc
WHEN NOT MATCHED THEN
    INSERT (
        weather_station_id,
        name,
        country,
        region,
        latitude,
        longitude,
        elevation_meters,
        timezone,
        distance_from_central_park_km,
        priority_rank,
        load_timestamp_utc
    )
    VALUES (
        source.weather_station_id,
        source.name,
        source.country,
        source.region,
        source.latitude,
        source.longitude,
        source.elevation_meters,
        source.timezone,
        source.distance_from_central_park_km,
        source.priority_rank,
        source.load_timestamp_utc
    );
