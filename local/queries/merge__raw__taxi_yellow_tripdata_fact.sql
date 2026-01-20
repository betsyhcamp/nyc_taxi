-- STAGING VIEW
CREATE OR REPLACE TEMP VIEW staging_yellow AS
SELECT
    *,
    regexp_extract(filename, '.*/([^/]+)$', 1) AS sourcefile_name
FROM read_parquet({{parquet_files}}, filename=true)
WHERE
    (
        {{year_month_start}} IS NULL
        OR regexp_extract(filename, '([0-9]{4}-[0-9]{2})', 1) >= {{year_month_start}}
    )
    AND (
        {{year_month_end_inclusive}} IS NULL
        OR regexp_extract(filename, '([0-9]{4}-[0-9]{2})', 1) <= {{year_month_end_inclusive}}
    );

-- MERGE STATEMENT

MERGE INTO raw.taxi_yellow_tripdata_fact AS target
USING (
    WITH staged AS (
        SELECT
            LOWER(
                MD5(
                    printf(
                        '%s|%s|%s|%s|%s',
                        COALESCE(CAST(vendorid AS VARCHAR), ''),
                        COALESCE(strftime(tpep_pickup_datetime, '%Y-%m-%d %H:%M:%S'), ''),
                        COALESCE(CAST(pulocationid AS VARCHAR), ''),
                        COALESCE(CAST(dolocationid AS VARCHAR), ''),
                        COALESCE(sourcefile_name, '')
                    )
                )
            ) AS trip_id,
            sourcefile_name,
            current_timestamp AS load_timestamp_utc,
            vendorid,
            tpep_pickup_datetime,
            tpep_dropoff_datetime,
            passenger_count,
            trip_distance,
            ratecodeid,
            store_and_fwd_flag,
            pulocationid,
            dolocationid,
            payment_type,
            fare_amount,
            extra,
            mta_tax,
            tip_amount,
            tolls_amount,
            improvement_surcharge,
            total_amount,
            congestion_surcharge,
            cbd_congestion_fee,
            airport_fee
        FROM staging_yellow
    ),
    deduped AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY trip_id
                ORDER BY tpep_dropoff_datetime DESC NULLS LAST, tpep_pickup_datetime DESC NULLS LAST
            ) AS rn
        FROM staged
    )
    SELECT
        trip_id,
        sourcefile_name,
        load_timestamp_utc,
        vendorid,
        tpep_pickup_datetime,
        tpep_dropoff_datetime,
        passenger_count,
        trip_distance,
        ratecodeid,
        store_and_fwd_flag,
        pulocationid,
        dolocationid,
        payment_type,
        fare_amount,
        extra,
        mta_tax,
        tip_amount,
        tolls_amount,
        improvement_surcharge,
        total_amount,
        congestion_surcharge,
        cbd_congestion_fee,
        airport_fee
    FROM deduped
    WHERE rn = 1
) AS source
ON target.trip_id = source.trip_id
WHEN MATCHED THEN
    UPDATE SET
        trip_id = source.trip_id,
        vendorid = source.vendorid,
        tpep_pickup_datetime = source.tpep_pickup_datetime,
        tpep_dropoff_datetime = source.tpep_dropoff_datetime,
        passenger_count = source.passenger_count,
        trip_distance = source.trip_distance,
        ratecodeid = source.ratecodeid,
        store_and_fwd_flag = source.store_and_fwd_flag,
        pulocationid = source.pulocationid,
        dolocationid = source.dolocationid,
        payment_type = source.payment_type,
        fare_amount = source.fare_amount,
        extra = source.extra,
        mta_tax = source.mta_tax,
        tip_amount = source.tip_amount,
        tolls_amount = source.tolls_amount,
        improvement_surcharge = source.improvement_surcharge,
        total_amount = source.total_amount,
        congestion_surcharge = source.congestion_surcharge,
        cbd_congestion_fee = source.cbd_congestion_fee,
        airport_fee = source.airport_fee,
        sourcefile_name = source.sourcefile_name,
        load_timestamp_utc = source.load_timestamp_utc
WHEN NOT MATCHED THEN
    INSERT (
        trip_id,
        vendorid,
        tpep_pickup_datetime,
        tpep_dropoff_datetime,
        passenger_count,
        trip_distance,
        ratecodeid,
        store_and_fwd_flag,
        pulocationid,
        dolocationid,
        payment_type,
        fare_amount,
        extra,
        mta_tax,
        tip_amount,
        tolls_amount,
        improvement_surcharge,
        total_amount,
        congestion_surcharge,
        airport_fee,
        cbd_congestion_fee,
        sourcefile_name,
        load_timestamp_utc
    )
    VALUES (
        source.trip_id,
        source.vendorid,
        source.tpep_pickup_datetime,
        source.tpep_dropoff_datetime,
        source.passenger_count,
        source.trip_distance,
        source.ratecodeid,
        source.store_and_fwd_flag,
        source.pulocationid,
        source.dolocationid,
        source.payment_type,
        source.fare_amount,
        source.extra,
        source.mta_tax,
        source.tip_amount,
        source.tolls_amount,
        source.improvement_surcharge,
        source.total_amount,
        source.congestion_surcharge,
        source.airport_fee,
        source.cbd_congestion_fee,
        source.sourcefile_name,
        source.load_timestamp_utc
    );

