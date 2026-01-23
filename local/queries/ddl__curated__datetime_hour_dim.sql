-- Hour-level datetime dimension table providing calendar attributes for each modeled hour.
-- Local Eastern Timezone (America/New_York).

CREATE SCHEMA IF NOT EXISTS curated;

CREATE TABLE IF NOT EXISTS curated.datetime_hour_dim (
    -- Start of the hour in local time truncated to the hour; primary key.
    datetime_hour TIMESTAMP NOT NULL,
    -- Calendar date corresponding to datetime_hour; FK to date_dim.calendar_date.
    calendar_date DATE NOT NULL,
    -- Hour of day in local time using 24-hour clock (0-23).
    hour_of_day BIGINT NOT NULL,
    -- Timestamp in UTC when row was loaded.
    load_timestamp_utc TIMESTAMP,
    PRIMARY KEY (datetime_hour)
);

