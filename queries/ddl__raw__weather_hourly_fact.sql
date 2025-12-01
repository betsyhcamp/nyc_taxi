CREATE OR REPLACE TABLE `nyc-taxi-ehc.raw.weather_hourly_fact`
(
    weather_station_id STRING NOT NULL
    OPTIONS (
        description = "Weather station identifier such as KNYC or KLGA; primary key component for the weather fact table."
    ),
    datetime_hour TIMESTAMP NOT NULL
    OPTIONS (
        description = "Hour start timestamp in local time for which the weather observations apply."
    ),
    calendar_date DATE NOT NULL
    OPTIONS (
        description = "Calendar date corresponding to datetime_hour; foreign key to date_dim calendar_date in ISO format YYYY-MM-DD"
    ),
    temp_celcius FLOAT64
    OPTIONS (
        description = "Air temperature in degrees Celsius."
    ),
    precipitation_millimeters FLOAT64
    OPTIONS (
        description = "Total liquid precipitation in millimeters recorded during the hour."
    ),
    snowfall_centimeters FLOAT64
    OPTIONS (
        description = "Total snowfall in centimeters recorded during the hour."
    ),
    wind_direction_degrees FLOAT64
    OPTIONS (
        description = "Wind direction in degrees where 0 means north and values increase clockwise."
    ),
    wind_speed_kilometers_per_hour FLOAT64
    OPTIONS (
        description = "Sustained wind speed in kilometers per hour."
    ),
    wind_gust_kilometers_per_hour FLOAT64
    OPTIONS (
        description = "Maximum wind gust speed in kilometers per hour during the hour."
    ),
    PRIMARY KEY (weather_station_id, datetime_hour) NOT ENFORCED
)
PARTITION BY calendar_date
CLUSTER BY weather_station_id, datetime_hour
OPTIONS (
    description = "Hourly weather fact table keyed by station and hour, used to join meteorological features to taxi demand facts."
);