SET VARIABLE zone_shape_zip = {{zone_shape_zip}};

INSTALL spatial;
LOAD spatial;

-- STAGING VIEW -- addresses a duplicate key which was causing merge to fail
CREATE OR REPLACE TEMP VIEW staging_taxi_zone_centroids AS
SELECT
    -- NOTE: The TLC taxi zone shapefile can contain multiple geometry parts for the same
    -- LocationID (e.g., discontiguous polygons, islands, or multipart shapes). If we compute
    -- centroids per-row, we can end up with duplicate taxi_zone_id entries in the staging data
    -- even though the dimension expects a single row per zone. To avoid 
    -- arbitrarily picking one geometry, we first union all geometry parts per LocationID
    -- and then compute a single centroid from that unified geometry.
    CAST(locationid AS BIGINT) AS taxi_zone_id,
    st_y(st_centroid(st_union_agg(geom))) AS centroid_latitude,
    st_x(st_centroid(st_union_agg(geom))) AS centroid_longitude
FROM st_read(getvariable('zone_shape_zip'))
GROUP BY locationid;

-- MERGE STATEMENT

MERGE INTO staging.taxi_zone_centroids_dim AS target
USING (
    WITH deduped AS (
        SELECT
            taxi_zone_id,
            centroid_latitude,
            centroid_longitude,
            ROW_NUMBER() OVER (
                PARTITION BY taxi_zone_id
                ORDER BY taxi_zone_id
            ) AS rn
        FROM staging_taxi_zone_centroids
    )
    SELECT
        taxi_zone_id,
        centroid_latitude,
        centroid_longitude
    FROM deduped
    WHERE rn = 1
) AS source
ON target.taxi_zone_id = source.taxi_zone_id
WHEN MATCHED THEN
    UPDATE SET
        centroid_latitude = source.centroid_latitude,
        centroid_longitude = source.centroid_longitude
WHEN NOT MATCHED THEN
    INSERT (
        taxi_zone_id,
        centroid_latitude,
        centroid_longitude
    )
    VALUES (
        source.taxi_zone_id,
        source.centroid_latitude,
        source.centroid_longitude
    );

