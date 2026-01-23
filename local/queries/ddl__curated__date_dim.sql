-- Date dimension table with calendar attributes, holiday and daylight savings flags.

CREATE SCHEMA IF NOT EXISTS curated;

CREATE TABLE IF NOT EXISTS curated.date_dim (
    -- Calendar date in YYYY-MM-DD format. Primary key for the date dimension.
    calendar_date DATE NOT NULL,
    -- 4-digit calendar year (e.g., 2019).
    calendar_year BIGINT NOT NULL,
    -- Calendar month number (1-12).
    calendar_month BIGINT NOT NULL,
    -- Full month name (e.g., 'January').
    month_name VARCHAR NOT NULL,
    -- Day of month (1-31).
    calendar_day BIGINT NOT NULL,
    -- Integer representation of day of week (1-7). Convention: 1=Monday.
    day_of_week BIGINT NOT NULL,
    -- Name of the day of week (e.g., 'Monday').
    day_of_week_name VARCHAR NOT NULL,
    -- TRUE if the date falls on a weekend (Saturday or Sunday).
    is_weekend BOOLEAN NOT NULL,
    -- TRUE if the date is a public holiday according to the configured holidays calendar.
    is_holiday BOOLEAN NOT NULL,
    -- Name of the holiday if is_holiday = TRUE; otherwise NULL.
    holiday_name VARCHAR,
    -- Country code for the holiday (e.g., 'US'). May be NULL if not applicable.
    holiday_country_code VARCHAR,
    -- TRUE if the date is during daylight savings in America/New_York timezone.
    is_daylight_savings BOOLEAN,
    -- Rule set used to generate holidays as country-subdivision-holiday_type-version.
    holiday_calendar_version VARCHAR,
    -- Timestamp in UTC when row was loaded.
    load_timestamp_utc TIMESTAMP,
    PRIMARY KEY (calendar_date)
);

