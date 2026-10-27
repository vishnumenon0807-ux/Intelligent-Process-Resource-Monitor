-- ===================================================================
-- iprm_forecast.sql
-- Wires the LSTM into the rest of the system.
--
-- Run once, in pgAdmin, against the iprm database.
-- Safe to re-run: every statement is guarded.
-- ===================================================================


-- -------------------------------------------------------------------
-- 1. The predictions table
-- -------------------------------------------------------------------
-- Two timestamps, and that is the whole point of the design.
--
--   made_at    when the model spoke
--   target_ts  the moment it was speaking about
--
-- Storing both means a forecast can be scored once its moment has
-- passed. With only one timestamp you can draw a forecast line but
-- never prove it was right.
--
-- Long format, one row per metric, so adding a fourth target later
-- needs no migration.

CREATE TABLE IF NOT EXISTS predictions (
    id         bigserial PRIMARY KEY,
    model_id   bigint NOT NULL REFERENCES model_registry(id),
    made_at    timestamptz NOT NULL DEFAULT now(),
    target_ts  timestamptz NOT NULL,
    horizon_s  integer NOT NULL,
    metric     text NOT NULL,
    predicted  double precision NOT NULL
);

-- The forecast chart queries by metric over a time range, newest first.
CREATE INDEX IF NOT EXISTS idx_predictions_metric_target
    ON predictions (metric, target_ts DESC);

-- The accuracy query scans predictions whose moment has passed.
CREATE INDEX IF NOT EXISTS idx_predictions_target_ts
    ON predictions (target_ts);


-- -------------------------------------------------------------------
-- 2. Forecast against reality
-- -------------------------------------------------------------------
-- Each prediction paired with the snapshot closest to the moment it
-- was predicting. `actual` stays NULL until that moment arrives, which
-- is exactly right: those rows are the future part of the chart.
--
-- The 3-second tolerance absorbs the gap between the collector's clock
-- and the horizon arithmetic. Widen it if the collector interval
-- changes.

CREATE OR REPLACE VIEW v_prediction_accuracy AS
SELECT p.id,
       p.model_id,
       p.metric,
       p.made_at,
       p.target_ts,
       p.horizon_s,
       p.predicted,
       CASE p.metric
           WHEN 'cpu_percent' THEN s.cpu_percent
           WHEN 'ram_percent' THEN s.ram_percent
           WHEN 'gpu_percent' THEN s.gpu_percent
       END AS actual
FROM predictions p
LEFT JOIN LATERAL (
    SELECT cpu_percent, ram_percent, gpu_percent
    FROM system_snapshots s
    WHERE s.timestamp BETWEEN p.target_ts - interval '3 seconds'
                          AND p.target_ts + interval '3 seconds'
    ORDER BY abs(extract(epoch FROM (s.timestamp - p.target_ts)))
    LIMIT 1
) s ON true;


-- -------------------------------------------------------------------
-- 3. Checks
-- -------------------------------------------------------------------
-- Run these after predictor.py has been going for a few minutes.

-- Seven columns, not one. If this returns a single `name` column, an
-- old stub table is still in place: DROP TABLE predictions CASCADE
-- and re-run this file.
--   SELECT column_name, data_type
--   FROM information_schema.columns
--   WHERE table_name = 'predictions'
--   ORDER BY ordinal_position;

-- Forecasts are landing.
--   SELECT metric, count(*), min(target_ts), max(target_ts)
--   FROM predictions GROUP BY metric;

-- Rolling accuracy on data the model has never seen. This is a better
-- number than the training RMSE, and it is the one worth quoting.
--   SELECT metric,
--          round(sqrt(avg(power(predicted - actual, 2)))::numeric, 2) AS rmse,
--          round(avg(abs(predicted - actual))::numeric, 2) AS mae,
--          count(*) AS scored
--   FROM v_prediction_accuracy
--   WHERE actual IS NOT NULL
--     AND made_at > now() - interval '24 hours'
--   GROUP BY metric;


-- -------------------------------------------------------------------
-- 4. The foreign key that was left commented out
-- -------------------------------------------------------------------
-- advisories.anomaly_id was the only unenforced relationship in the
-- schema. Adding it here while the constraints are already open.
--
-- It will fail if any advisory points at an anomaly that no longer
-- exists. Check first:
--
--   SELECT count(*) FROM advisories a
--   WHERE a.anomaly_id IS NOT NULL
--     AND NOT EXISTS (SELECT 1 FROM anomalies n WHERE n.id = a.anomaly_id);
--
-- Zero means the line below is safe. Anything else needs those rows
-- cleaned up first, so this stays commented until you have checked.

-- ALTER TABLE advisories
--     ADD CONSTRAINT fk_advisories_anomaly
--     FOREIGN KEY (anomaly_id) REFERENCES anomalies(id) ON DELETE SET NULL;


-- -------------------------------------------------------------------
-- 5. Dedupe index
-- -------------------------------------------------------------------
-- Matches the already_advised() fix that keys on process name rather
-- than PID. Without this index the dedupe check scans the advisories
-- table on every write.

CREATE INDEX IF NOT EXISTS idx_advisories_dedupe
    ON advisories (lower(target_name), rule_id, created_at DESC);
