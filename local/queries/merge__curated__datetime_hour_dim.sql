-- Merge hourly datetime dimension for a specified date range.
-- Generates one row per hour from start_date 00:00 through end_date 23:00.

SET VARIABLE start_date = {{start_date}};
SET VARIABLE end_date = {{end_date}};

MERGE INTO curated.datetime_hour_dim AS target
USING (
    SELECT
        datetime_hour,
        CAST(datetime_hour AS DATE) AS calendar_date,
        EXTRACT(hour FROM datetime_hour) AS hour_of_day,
        current_timestamp AS load_timestamp_utc
    FROM generate_series(
        CAST(getvariable('start_date') AS TIMESTAMP),
        CAST(getvariable('end_date') AS TIMESTAMP) + INTERVAL '23 hours',
        INTERVAL 1 HOUR
    ) AS t(datetime_hour)
) AS source
ON target.datetime_hour = source.datetime_hour
WHEN MATCHED THEN
    UPDATE SET
        calendar_date = source.calendar_date,
        hour_of_day = source.hour_of_day,
        load_timestamp_utc = source.load_timestamp_utc
WHEN NOT MATCHED THEN
    INSERT (
        datetime_hour,
        calendar_date,
        hour_of_day,
        load_timestamp_utc
    )
    VALUES (
        source.datetime_hour,
        source.calendar_date,
        source.hour_of_day,
        source.load_timestamp_utc
    );
