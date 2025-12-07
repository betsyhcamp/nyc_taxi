CREATE OR REPLACE VIEW `nyc-taxi-ehc.curated.taxi_zone_dim` 
OPTIONS (
  description = "Curated taxi zone dimension including borough, zone name, service zone, and centroid coordinates joined from staging.taxi_zone_centroids_dim."
) AS
  SELECT
        z.LocationID AS taxi_zone_id,
        z.Borough AS borough,
        z.Zone AS zone_name,
        z.service_zone,
        c.centroid_latitude,
        c.centroid_longitude
    FROM `nyc-taxi-ehc.raw.taxi_zone_dim` AS z
    LEFT JOIN `nyc-taxi-ehc.staging.taxi_zone_centroids_dim` AS c
        ON z.LocationID = c.taxi_zone_id