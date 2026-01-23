-- Weather station dimension table with metadata for stations used in weather fact tables.

CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS raw.weather_station_dim (
    -- Meteostat weather station identifier (e.g., 'KNYC0', 'KTEB0', '72502').
    weather_station_id VARCHAR NOT NULL,
    -- Human-readable station name (e.g., 'New York City / Yorkville').
    name VARCHAR,
    -- ISO 3166-1 alpha-2 country code (e.g., 'US').
    country VARCHAR,
    -- ISO 3166-2 state or region code (e.g., 'NY', 'NJ').
    region VARCHAR,
    -- Station latitude in decimal degrees.
    latitude DOUBLE,
    -- Station longitude in decimal degrees.
    longitude DOUBLE,
    -- Station elevation in meters above sea level.
    elevation_meters INTEGER,
    -- IANA timezone name (e.g., 'America/New_York').
    timezone VARCHAR,
    -- Distance from Central Park (40.7829, -73.9654) in kilometers.
    distance_from_central_park_km DOUBLE,
    -- Priority rank for coalescing (1 = highest priority, used first).
    priority_rank INTEGER,
    -- Timestamp in UTC when row was loaded.
    load_timestamp_utc TIMESTAMP,
    PRIMARY KEY (weather_station_id)
);
