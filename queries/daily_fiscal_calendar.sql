-- One row per date, the fiscal calendar's source. No ride data bounds it, so the
-- calendar runs past the panel.

SELECT
    calendar_date,
    fiscal_week_of_month,
    fiscal_month,
    fiscal_year,
    fiscal_year_month,
    is_weekend,
    is_holiday,
    DATE_TRUNC(calendar_date, WEEK (SUNDAY)) AS fiscal_week_start_date,
    (fiscal_year * 100) + fiscal_week AS fiscal_year_week
FROM `nyc-taxi-ehc.curated.date_dim`
-- A month is provably complete only when the table holds a later date; a partial
-- one undercounts weeks_in_month and count_workdays.
WHERE
    fiscal_year_month < (
        SELECT MAX(latest.fiscal_year_month)
        FROM `nyc-taxi-ehc.curated.date_dim` AS latest
    )
ORDER BY calendar_date
