CREATE OR REPLACE TABLE `nyc-taxi-ehc.raw.taxi_zone_dim`
(
    LocationID INT64 NOT NULL
    OPTIONS (
        description = "Primary key for the taxi zone. Matches LocationID from NYC TLC lookup."
    ),
    Borough STRING
    OPTIONS (
        description = "NYC borough name (e.g., 'Manhattan', 'Queens') as provided by the NYC TLC lookup."
    ),
    Zone STRING
    OPTIONS (
        description = "Human-readable zone name (e.g., 'Midtown Center')."
    ),
    service_zone STRING
    OPTIONS (
        description = "Service zone category from NYC TLC (e.g., 'yellow', 'green', 'Boro Zone'). May be NULL."
    ),
    load_timestamp_utc TIMESTAMP 
    OPTIONS (
        description = "Timestamp in UTC when row was loaded."
    ),
    PRIMARY KEY (LocationID) NOT ENFORCED
)
CLUSTER BY Borough, service_zone
OPTIONS (
    description = "Taxi zone dimension table from the NYC Taxi and Limousine Commission (TLC) zone lookup file."
);