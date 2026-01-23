-- Staging table for taxi zone centroids computed from TLC shapefile geometries.

CREATE SCHEMA IF NOT EXISTS staging;

CREATE TABLE IF NOT EXISTS staging.taxi_zone_centroids_dim (
    -- TLC taxi zone location ID; primary key.
    taxi_zone_id BIGINT NOT NULL,
    -- Centroid latitude in decimal degrees.
    centroid_latitude DOUBLE,
    -- Centroid longitude in decimal degrees.
    centroid_longitude DOUBLE,
    -- Timestamp in UTC when row was loaded.
    load_timestamp_utc TIMESTAMP,
    PRIMARY KEY (taxi_zone_id)
);

