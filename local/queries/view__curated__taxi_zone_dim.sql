CREATE OR REPLACE TABLE curated.taxi_zone_dim AS
SELECT
    z.locationid AS taxi_zone_id,
    z.borough,
    z.zone AS zone_name,
    z.service_zone,
    c.centroid_latitude,
    c.centroid_longitude
FROM raw.taxi_zone_dim AS z
LEFT JOIN staging.taxi_zone_centroids_dim AS c
    ON z.locationid = c.taxi_zone_id;

