CREATE OR REPLACE TABLE `nyc-taxi-ehc.curated.datetime_hour_dim`
(
    datetime_hour TIMESTAMP NOT NULL
    OPTIONS (
        description = "Start of the hour (e.g., 2018-01-01 13:00:00-05:00) in the NYC local timezone. Primary key."
    ),
    calendar_date DATE NOT NULL
    OPTIONS (
        description = "Calendar date corresponding to datetime_hour. Foreign key to dim_date.date in ISO format YYYY-MM-DD"
    ),
    PRIMARY KEY (datetime_hour) NOT ENFORCED
)
PARTITION BY DATE(datetime_hour)
CLUSTER BY year, month, day_of_week
OPTIONS (
    description = "Hour-level datetime dimension table providing calendar attributes for each modeled hour."
);
