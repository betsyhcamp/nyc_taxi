CREATE OR REPLACE TABLE `nyc-taxi-ehc.raw.taxi_yellow_tripdata_fact`
(
    VendorID INT64
    OPTIONS (
        description = "TLC vendor ID for the taxi company (e.g., 1 or 2)."
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
    RatecodeID INT64
    OPTIONS (
        description = "Rate code in effect at the end of the trip (e.g., standard rate, JFK, Newark)."
    ),
    store_and_fwd_flag STRING
    OPTIONS (
        description = "Flag indicating whether the trip record was held in-vehicle before sending to the vendor ('Y' or 'N')."
    ),
    PULocationID INT64
    OPTIONS (
        description = "TLC taxi zone ID where the passenger was picked up (PULocationID)."
    ),
    DOLocationID INT64
    OPTIONS (
        description = "TLC taxi zone ID where the passenger was dropped off (DOLocationID)."
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
        description = "MTA tax imposed per trip in USD, typically a fixed amount."
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
        description = "Total amount charged to the passenger in USD, including all surcharges, tolls, and tip."
    ),
    congestion_surcharge NUMERIC
    OPTIONS (
        description = "NYC congestion surcharge in USD, if applicable."
    ),
    Airport_fee NUMERIC
    OPTIONS (
        description = "Airport access fee in USD, typically for pickups at eligible airports."
    ),
    sourcefile_name STRING NOT NULL
    OPTIONS (
        description = "Original source file name (e.g., 'yellow_tripdata_2018-01.parquet' or similar) for traceability."
    )
)
PARTITION BY DATE(tpep_pickup_datetime)
CLUSTER BY PULocationID, DOLocationID, tpep_pickup_datetime
OPTIONS (
    description = "Raw yellow taxi trip fact table loaded from NYC TLC files, preserving original TLC schema and adding only a sourcefile_name column for lineage."
);