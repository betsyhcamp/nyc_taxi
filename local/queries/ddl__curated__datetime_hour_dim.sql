CREATE SCHEMA IF NOT EXISTS curated;

CREATE OR REPLACE TABLE curated.datetime_hour_dim (
    datetime_hour TIMESTAMP NOT NULL,
    calendar_date DATE NOT NULL,
    hour_of_day BIGINT NOT NULL,
    load_timestamp_utc TIMESTAMP,
    PRIMARY KEY (datetime_hour)
);

