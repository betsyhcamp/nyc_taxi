CREATE SCHEMA IF NOT EXISTS curated;

CREATE OR REPLACE TABLE curated.date_dim (
    calendar_date DATE NOT NULL,
    calendar_year BIGINT NOT NULL,
    calendar_month BIGINT NOT NULL,
    month_name VARCHAR NOT NULL,
    calendar_day BIGINT NOT NULL,
    day_of_week BIGINT NOT NULL,
    day_of_week_name VARCHAR NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    is_holiday BOOLEAN NOT NULL,
    holiday_name VARCHAR,
    holiday_country_code VARCHAR,
    is_daylight_savings BOOLEAN,
    holiday_calendar_version VARCHAR,
    load_timestamp_utc TIMESTAMP,
    PRIMARY KEY (calendar_date)
);

