CREATE OR REPLACE TABLE `nyc-taxi-ehc.curated.datetime_hour_dim`
(
    datetime_hour TIMESTAMP NOT NULL
    OPTIONS (
        description = "Start of the hour in local time (e.g., America/New_York) truncated to the hour; primary key."
    ),
    calendar_date DATE NOT NULL
    OPTIONS (
        description = "Calendar date corresponding to datetime_hour. Foreign key to date_dim.calendar_date in ISO format YYYY-MM-DD"
    ),
    hour_of_day INT64 NOT NULL
    OPTIONS (
        description = "Hour of day in local time using 24-hour clock (0–23)."
    ),
    load_timestamp_utc TIMESTAMP 
    OPTIONS (
        description = "Timestamp in UTC when row was loaded."
    ),
    PRIMARY KEY (datetime_hour) NOT ENFORCED
)
PARTITION BY DATE(datetime_hour)
CLUSTER BY calendar_date,hour_of_day
OPTIONS (
    description = "Hour-level datetime dimension table providing calendar attributes for each modeled hour. Local Eastern Timezone."
);
