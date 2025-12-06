CREATE OR REPLACE TABLE `nyc-taxi-ehc.raw.weather_daily_fact`
(
    weather_station_id STRING NOT NULL
    OPTIONS (
        description = "Weather station identifier (e.g., 'KNYC', 'KLGA') from the original weather source."
    ),
    calendar_date DATE NOT NULL
    OPTIONS (
        description = "Calendar date for the daily aggregated weather metrics; foreign key to date_dim.calendar_date in ISO format YYYY-MM-DD."
    ),
    avg_temp_celcius FLOAT64
    OPTIONS (
        description = "Average air temperature in degrees Celsius over the full day."
    ),
    min_temp_celcius FLOAT64
    OPTIONS (
        description = "Minimum hourly air temperature in degrees Celsius observed during the day."
    ),
    max_temp_celcius FLOAT64
    OPTIONS (
        description = "Maximum hourly air temperature in degrees Celsius observed during the day."
    ),
    total_precip_millimeters FLOAT64
    OPTIONS (
        description = "Total precipitation in millimeters accumulated during the day."
    ),
    total_snow_centimeters FLOAT64
    OPTIONS (
        description = "Total snowfall in centimeters accumulated during the day."
    ),
    avg_wind_speed_kilometers_per_hour FLOAT64
    OPTIONS (
        description = "Average wind speed in kilometers per hour across all hourly observations during the day."
    ),
    max_wind_gust_kilometers_per_hour FLOAT64
    OPTIONS (
        description = "Maximum wind gust in kilometers per hour observed during the day, if available."
    ),
    load_timestamp_utc TIMESTAMP 
    OPTIONS (
        description = "Timestamp in UTC when row was loaded."
    ),
    PRIMARY KEY (weather_station_id, calendar_date) NOT ENFORCED
)
PARTITION BY calendar_date
CLUSTER BY weather_station_id
OPTIONS (
    description = "Daily weather fact table keyed by station and date, providing aggregated temperature, precipitation, snowfall, humidity, and wind metrics for modeling lower-frequency effects."
);

