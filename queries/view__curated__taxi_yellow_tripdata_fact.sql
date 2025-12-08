CREATE OR REPLACE VIEW `nyc-taxi-ehc.curated.taxi_yellow_tripdata_fact` AS 
(
  SELECT
  -- hash for trip_id
  TO_HEX(
    MD5(
      FORMAT(
        '%d|%s|%d|%d|%s',
        vendorid,
        FORMAT_TIMESTAMP('%F %T%Ez', tpep_pickup_datetime),
        pulocationid,
        dolocationid,
        sourcefile_name
      )
    )
  ) AS trip_id,

  --original columns w/ some renamed for clarity
  vendorid AS vendor_id,
  tpep_pickup_datetime,
  tpep_dropoff_datetime,
  passenger_count,
  trip_distance,
  ratecodeid AS ratecode_id,
  store_and_fwd_flag,
  pulocationid AS pickup_taxi_zone_id,
  dolocationid AS dropoff_taxi_zone_id,
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

  --Additional date and time columns relating to the ride
  DATE(tpep_pickup_datetime, 'America/New_York') AS pickup_date,
  TIMESTAMP_TRUNC(tpep_pickup_datetime, HOUR, 'America/New_York') AS pickup_datetime_hour,
  DATE(tpep_dropoff_datetime, 'America/New_York') AS dropoff_date,
  TIMESTAMP_TRUNC(tpep_dropoff_datetime, HOUR, 'America/New_York') AS dropoff_datetime_hour,
  
  sourcefile_name
  
  FROM `nyc-taxi-ehc.staging.taxi_yellow_tripdata_fact_silver`
)