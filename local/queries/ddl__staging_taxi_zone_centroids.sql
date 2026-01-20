CREATE SCHEMA IF NOT EXISTS staging;

CREATE OR REPLACE TABLE staging.taxi_zone_centroids_dim (
    taxi_zone_id BIGINT NOT NULL,
    centroid_latitude DOUBLE,
    centroid_longitude DOUBLE,
    PRIMARY KEY (taxi_zone_id)
);

