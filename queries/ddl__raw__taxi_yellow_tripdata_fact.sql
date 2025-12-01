CREATE OR REPLACE TABLE `nyc-taxi-ehc.raw.taxi_yellow_tripdata_fact`
(
    trip_id STRING NOT NULL
    OPTIONS (
        description = "Surrogate primary key for a taxi trip, typically generated as a stable hash over key raw fields."
    ),
    vendor_id INT64
    OPTIONS (
        description = "TLC assigned vendor code indicating the taxi company or technology provider."
    ),
    tpep_pickup_datetime TIMESTAMP NOT NULL
    OPTIONS (
        description = "Trip pickup timestamp from the yellow taxi data in local New York time."
    ),
    tpep_dropoff_datetime TIMESTAMP NOT NULL
    OPTIONS (
        description = "Trip dropoff timestamp from the yellow taxi data in local New York time."
    ),
    pickup_date DATE NOT NULL
    OPTIONS (
        description = "Calendar date corresponding to tpep_pickup_datetime; used for partitioning."
    ),
    pickup_datetime_hour TIMESTAMP NOT NULL
    OPTIONS (
        description = "Pickup timestamp truncated to the hour; foreign key to dim_datetime_hour.datetime_hour."
    ),
    pickup_taxi_zone_id INT64 NOT NULL
    OPTIONS (
        description = "Pickup zone identifier (LocationID) from the TLC data; foreign key to dim_taxi_zone.taxi_zone_id."
    ),
    dropoff_taxi_zone_id INT64
    OPTIONS (
        description = "Dropoff zone identifier (LocationID) from the TLC data; foreign key to dim_taxi_zone.taxi_zone_id."
    ),
    passenger_count INT64
    OPTIONS (
        description = "Number of passengers reported on the trip."
    ),
    trip_distance FLOAT64
    OPTIONS (
        description = "Trip distance measured in miles as reported by the taximeter."
    ),
    rate_code_id INT64
    OPTIONS (
        description = "TLC rate code identifier describing the pricing category for the trip."
    ),
    store_and_fwd_flag STRING
    OPTIONS (
        description = "Indicator that the trip record was held in vehicle memory before sending to the vendor."
    ),
    payment_type INT64
    OPTIONS (
        description = "Numeric code representing the payment method (for example, credit card or cash)."
    ),
    fare_amount NUMERIC
    OPTIONS (
        description = "Base fare amount for the trip in USD."
    ),
    extra NUMERIC
    OPTIONS (
        description = "Miscellaneous extras and surcharges in USD."
    ),
    mta_tax NUMERIC
    OPTIONS (
        description = "MTA tax amount in USD."
    ),
    tip_amount NUMERIC
    OPTIONS (
        description = "Tip amount in USD."
    ),
    tolls_amount NUMERIC
    OPTIONS (
        description = "Total tolls paid in USD."
    ),
    improvement_surcharge NUMERIC
    OPTIONS (
        description = "Improvement surcharge amount in USD."
    ),
    congestion_surcharge NUMERIC
    OPTIONS (
        description = "Congestion surcharge amount in USD where applicable."
    ),
    total_amount NUMERIC
    OPTIONS (
        description = "All in total amount for the trip in USD."
    ),
    source_file_name STRING
    OPTIONS (
        description = "NYC TLC source filename"
    ),
    load_timestamp_utc TIMESTAMP NOT NULL
    OPTIONS (
        description = "Timestamp when data was loaded in UTC"
    ),
    PRIMARY KEY (trip_id) NOT ENFORCED
)
PARTITION BY pickup_date
CLUSTER BY pickup_taxi_zone_id, pickup_datetime_hour
OPTIONS (
    description = "Raw yellow taxi trip fact table with one row per trip, derived from raw yellow_tripdata files and enriched with keys for joins to dimensions."
);