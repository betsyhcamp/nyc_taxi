SET VARIABLE start_date = {{start_date}};
SET VARIABLE end_date = {{end_date}};

MERGE INTO curated.date_dim AS target
USING (
    WITH dates AS (
        SELECT
            *
        FROM generate_series(
            CAST(getvariable('start_date') AS DATE),
            CAST(getvariable('end_date') AS DATE),
            INTERVAL 1 DAY
        ) AS t(calendar_date)
    )
    SELECT
        calendar_date,
        EXTRACT(year FROM calendar_date) AS calendar_year,
        EXTRACT(month FROM calendar_date) AS calendar_month,
        STRFTIME(calendar_date, '%B') AS month_name,
        EXTRACT(day FROM calendar_date) AS calendar_day,
        EXTRACT(isodow FROM calendar_date) AS day_of_week,
        STRFTIME(calendar_date, '%A') AS day_of_week_name,
        CASE
            WHEN EXTRACT(isodow FROM calendar_date) IN (6, 7) THEN TRUE
            ELSE FALSE
        END AS is_weekend,
        FALSE AS is_holiday,
        CAST(NULL AS VARCHAR) AS holiday_name,
        CAST(NULL AS VARCHAR) AS holiday_country_code,
        CAST(NULL AS BOOLEAN) AS is_daylight_savings,
        CAST(NULL AS VARCHAR) AS holiday_calendar_version,
        current_timestamp AS load_timestamp_utc
    FROM dates
) AS source
ON target.calendar_date = source.calendar_date
WHEN MATCHED THEN
    UPDATE SET
        calendar_year = source.calendar_year,
        calendar_month = source.calendar_month,
        month_name = source.month_name,
        calendar_day = source.calendar_day,
        day_of_week = source.day_of_week,
        day_of_week_name = source.day_of_week_name,
        is_weekend = source.is_weekend,
        is_holiday = source.is_holiday,
        holiday_name = source.holiday_name,
        holiday_country_code = source.holiday_country_code,
        is_daylight_savings = source.is_daylight_savings,
        holiday_calendar_version = source.holiday_calendar_version,
        load_timestamp_utc = source.load_timestamp_utc
WHEN NOT MATCHED THEN
    INSERT (
        calendar_date,
        calendar_year,
        calendar_month,
        month_name,
        calendar_day,
        day_of_week,
        day_of_week_name,
        is_weekend,
        is_holiday,
        holiday_name,
        holiday_country_code,
        is_daylight_savings,
        holiday_calendar_version,
        load_timestamp_utc
    )
    VALUES (
        source.calendar_date,
        source.calendar_year,
        source.calendar_month,
        source.month_name,
        source.calendar_day,
        source.day_of_week,
        source.day_of_week_name,
        source.is_weekend,
        source.is_holiday,
        source.holiday_name,
        source.holiday_country_code,
        source.is_daylight_savings,
        source.holiday_calendar_version,
        source.load_timestamp_utc
    );

