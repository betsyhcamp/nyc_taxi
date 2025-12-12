CREATE OR REPLACE TABLE `nyc-taxi-ehc.raw.taxi_zone_dim`
(
    LOCATIONID INT64 NOT NULL
    OPTIONS (
        description = "Primary key for the taxi zone. Matches LocationID from NYC TLC lookup."
    ),
    BOROUGH STRING
    OPTIONS (
        description = "NYC borough name (e.g., 'Manhattan', 'Queens') as provided by the NYC TLC lookup."
    ),
    ZONE STRING
    OPTIONS (
        description = "Human-readable zone name (e.g., 'Midtown Center')."
    ),
    SERVICE_ZONE STRING
    OPTIONS (
        description = "Service zone category from NYC TLC (e.g., 'yellow', 'green', 'Boro Zone'). May be NULL."
    ),
    LOAD_TIMESTAMP_UTC TIMESTAMP
    OPTIONS (
        description = "Timestamp in UTC when row was loaded."
    ),
    PRIMARY KEY (LOCATIONID) NOT ENFORCED
)
CLUSTER BY BOROUGH, SERVICE_ZONE
OPTIONS (
    description = "Taxi zone dimension table from the NYC Taxi and Limousine Commission (TLC) zone lookup file."
);
