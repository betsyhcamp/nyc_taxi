CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS raw.taxi_zone_dim (
    locationid BIGINT NOT NULL,
    borough VARCHAR,
    zone VARCHAR,
    service_zone VARCHAR,
    load_timestamp_utc TIMESTAMP,
    PRIMARY KEY (locationid)
);

