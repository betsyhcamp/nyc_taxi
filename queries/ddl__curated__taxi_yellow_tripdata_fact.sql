CREATE OR REPLACE TABLE `nyc-taxi-ehc.curated.taxi_yellow_tripdata_fact`
(
    trip_id STRING NOT NULL
    OPTIONS (
        description = "Surrogate primary key for the trip, a hash of fields (vendor_id, pickup timestamp, pickup & dropoff locations, and sourcefile_name)."
    ),

    -- Original TLC fields with some renaming of columns for clarity & to match other tables
    vendor_id INT64
    OPTIONS (
        description = "TLC vendor ID for the taxi company, carried through from the raw TLC record."
    ),
    tpep_pickup_datetime TIMESTAMP NOT NULL
    OPTIONS (
        description = "Pickup datetime in local NYC time as provided by TLC (tpep_pickup_datetime)."
    ),
    tpep_dropoff_datetime TIMESTAMP
    OPTIONS (
        description = "Dropoff datetime in local NYC time as provided by TLC (tpep_dropoff_datetime)."
    ),
    passenger_count INT64
    OPTIONS (
        description = "Number of passengers reported by the driver for the trip."
    ),
    trip_distance FLOAT64
    OPTIONS (
        description = "Distance of the trip in miles as reported by the taximeter."
    ),
    ratecode_id INT64
    OPTIONS (
        description = "Rate code in effect at the end of the trip (e.g., standard rate, JFK, Newark)."
    ),
    store_and_fwd_flag STRING
    OPTIONS (
        description = "Flag indicating whether the trip record was held in-vehicle before sending to the vendor ('Y' or 'N')."
    ),
    pickup_taxi_zone_id INT64
    OPTIONS (
        description = "Taxi zone ID for pickup, equivalent to PULocationID; foreign key to taxi_zone_dim.taxi_zone_id."
    ),
    dropoff_taxi_zone_id INT64
    OPTIONS (
        description = "Taxi zone ID for dropoff, equivalent to DOLocationID; foreign key to taxi_zone_dim.taxi_zone_id."
    ),
    payment_type INT64
    OPTIONS (
        description = "Numeric code identifying the payment method (e.g., 1=credit card, 2=cash)."
    ),
    fare_amount NUMERIC
    OPTIONS (
        description = "Metered fare amount in USD, excluding surcharges, tolls, and tips."
    ),
    extra NUMERIC
    OPTIONS (
        description = "Miscellaneous extras and surcharges in USD (e.g., night surcharge, rush hour surcharge)."
    ),
    mta_tax NUMERIC
    OPTIONS (
        description = "MTA tax imposed per trip in USD."
    ),
    tip_amount NUMERIC
    OPTIONS (
        description = "Tip amount in USD, automatically populated for credit card payments."
    ),
    tolls_amount NUMERIC
    OPTIONS (
        description = "Total tolls paid in USD during the trip."
    ),
    improvement_surcharge NUMERIC
    OPTIONS (
        description = "Improvement surcharge in USD, typically a fixed amount per trip."
    ),
    total_amount NUMERIC
    OPTIONS (
        description = "Total amount charged to the passenger in USD, including all surcharges, tolls, and tips."
    ),
    congestion_surcharge NUMERIC
    OPTIONS (
        description = "NYC congestion surcharge in USD, if applicable."
    ),
    airport_fee NUMERIC
    OPTIONS (
        description = "Airport access fee in USD, if applicable."
    ),
    cbd_congestion_fee NUMERIC
    OPTIONS (
        description = "Per-trip charge for MTA's Congestion Relief Zone (starting Jan. 5, 2025)."
    ),
    -- Enriched temporal fields
    pickup_date DATE NOT NULL
    OPTIONS (
        description = "Calendar date of the pickup (derived from tpep_pickup_datetime); foreign key to dim_date.calendar_date."
    ),
    pickup_datetime_hour TIMESTAMP NOT NULL
    OPTIONS (
        description = "Pickup datetime truncated to the hour in local NYC time; foreign key to datetime_hour_dim.datetime_hour."
    ),
    dropoff_date DATE
    OPTIONS (
        description = "Calendar date of the dropoff (derived from tpep_dropoff_datetime)."
    ),
    dropoff_datetime_hour TIMESTAMP
    OPTIONS (
        description = "Dropoff datetime truncated to the hour in local NYC time."
    ),

    -- Lineage and load metadata
    sourcefile_name STRING NOT NULL
    OPTIONS (
        description = "Original source file name used to load this trip from the raw layer for lineage and debugging."
    ),
    refresh_timestamp_utc TIMESTAMP NOT NULL
    OPTIONS (
        description = "UTC timestamp when this curated record was last refreshed in BigQuery."
    ),

    PRIMARY KEY (trip_id) NOT ENFORCED
)
PARTITION BY pickup_date
CLUSTER BY pickup_taxi_zone_id, pickup_datetime_hour
OPTIONS (
    description = "Curated yellow taxi trip fact table enriched with surrogate trip IDs, date and hour breakouts, normalized taxi zone keys, and load metadata; built from nyc-taxi-ehc.raw.taxi_yellow_tripdata_fact."
);