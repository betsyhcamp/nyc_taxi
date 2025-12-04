CREATE OR REPLACE TABLE `nyc-taxi-ehc.curated.taxi_zone_dim`
(
    taxi_zone_id INT64 NOT NULL
    OPTIONS (
        description = "Primary key for the taxi zone. Matches LocationID from NYC TLC lookup."
    ),
    borough STRING NOT NULL
    OPTIONS (
        description = "NYC borough name (e.g., 'Manhattan', 'Queens') as provided by the NYC TLC lookup."
    ),
    zone_name STRING NOT NULL
    OPTIONS (
        description = "Human-readable zone name (e.g., 'Midtown Center')."
    ),
    service_zone STRING
    OPTIONS (
        description = "Service zone category from NYC TLC (e.g., 'yellow', 'green', 'Boro Zone'). May be NULL."
    ),
    centroid_latitude FLOAT64
    OPTIONS (
        description = "Approximate latitude of the zone centroid. Nullable; computed from NYC TLC shapefile (GIS)."
    ),
    centroid_longitude FLOAT64
    OPTIONS (
        description = "Approximate longitude of the zone centroid. Nullable; computed from NYC TLC shapefile (GIS)."
    ),
    load_timestamp_utc TIMESTAMP 
    OPTIONS (
        description = "Timestamp in UTC when row was loaded."
    ),
    PRIMARY KEY (taxi_zone_id) NOT ENFORCED
)
CLUSTER BY borough, service_zone
OPTIONS (
    description = "Taxi zone dimension table derived from the NYC Taxi and Limousine Commission (TLC) zone lookup file, with taxi zone centroid coordinates."
);