SET VARIABLE zone_lookup_csv = {{zone_lookup_csv}};

-- STAGING VIEW
CREATE OR REPLACE TEMP VIEW staging_taxi_zone_lookup AS
SELECT
    CAST(locationid AS BIGINT) AS locationid,
    borough,
    zone,
    service_zone,
    current_timestamp AS load_timestamp_utc
FROM read_csv_auto(getvariable('zone_lookup_csv'), header=true);

-- MERGE STATEMENT

MERGE INTO raw.taxi_zone_dim AS target
USING (
    SELECT
        locationid,
        borough,
        zone,
        service_zone,
        load_timestamp_utc
    FROM staging_taxi_zone_lookup
) AS source
ON target.locationid = source.locationid
WHEN MATCHED THEN
    UPDATE SET
        borough = source.borough,
        zone = source.zone,
        service_zone = source.service_zone,
        load_timestamp_utc = source.load_timestamp_utc
WHEN NOT MATCHED THEN
    INSERT (
        locationid,
        borough,
        zone,
        service_zone,
        load_timestamp_utc
    )
    VALUES (
        source.locationid,
        source.borough,
        source.zone,
        source.service_zone,
        source.load_timestamp_utc
    );

