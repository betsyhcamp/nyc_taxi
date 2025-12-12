CREATE OR REPLACE TABLE `nyc-taxi-ehc.staging.taxi_zone_centroids_dim`
(
    taxi_zone_id INT64 NOT NULL
    OPTIONS (
        description = "Uniquely identifies each taxi zone; foreign key to curated.taxi_zone_dim.taxi_zone_id."
    ),

    centroid_latitude FLOAT64
    OPTIONS (
        description = "Latitude of the geographic centroid of the taxi zone in WGS84, computed from geometry of NYC TLC-provided GIS .shapefile."
    ),

    centroid_longitude FLOAT64
    OPTIONS (
        description
        = "Longitude of the geographic centroid of the taxi zone in WGS84, computed from geometry of NYC TLC-provided GIS .shapefile."
    ),
    PRIMARY KEY (taxi_zone_id) NOT ENFORCED
)
OPTIONS (
    description
    = "Staging (silver layer) dimension table containing computed centroid coordinates for each NYC taxi zone. Populated from NYC TLC provided geometry; joined into the curated.taxi_zone_dim as enriched spatial attributes."
);
