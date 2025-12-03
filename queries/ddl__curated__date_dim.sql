CREATE OR REPLACE TABLE `nyc-taxi-ehc.curated.date_dim`
(
    calendar_date DATE NOT NULL
    OPTIONS (
        description = "Calendar date in YYYY-MM-DD format. Primary key for the date dimension."
    ),
    calendar_year INT64 NOT NULL
    OPTIONS (
        description = "4-digit calendar year (e.g., 2019)."
    ),
    calendar_month INT64 NOT NULL
    OPTIONS (
        description = "Calendar month number (1–12)."
    ),
    month_name STRING NOT NULL
    OPTIONS (
        description = "Full month name (e.g., 'January')."
    ),
    calendar_day INT64 NOT NULL
    OPTIONS (
        description = "Day of month (1–31)."
    ),
    day_of_week INT64 NOT NULL
    OPTIONS (
        description = "Integer representation of day of week (1–7). Convention is project-defined (e.g. 1=Monday)."
    ),
    day_of_week_name STRING NOT NULL
    OPTIONS (
        description = "Name of the day of week (e.g., 'Monday')."
    ),
    is_weekend BOOL NOT NULL
    OPTIONS (
        description = "TRUE if the date falls on a weekend according to project-defined logic."
    ),
    is_holiday BOOL NOT NULL
    OPTIONS (
        description = "TRUE if the date is a public holiday according to the configured holidays calendar."
    ),
    holiday_name STRING
    OPTIONS (
        description = "Name of the holiday if is_holiday = TRUE; otherwise NULL."
    ),
    holiday_country_code STRING
    OPTIONS (
        description = "Country code for the holiday (e.g., 'US'). May be NULL if not applicable."
    ),
    is_daylight_savings BOOL 
    OPTIONS (
        description = "TRUE if the date is a day for daylight savings; otherwise FALSE. In local New York/Eastern Time timezon."
    ),
    holiday_calendar_version STRING 
    OPTIONS (
        description = "Rule set used to generate holidays as country-subdivision-holiday_type-version."
    ),
    created_timestamp_utc TIMESTAMP 
    OPTIONS (
        description = "Timestamp in UTC when row was created."
    ),
    PRIMARY KEY (calendar_date) NOT ENFORCED
)
PARTITION BY calendar_date
CLUSTER BY calendar_year, calendar_month
OPTIONS (
    description = "Date dimension table with calendar attributes, with holiday and daylight savings flags."
);