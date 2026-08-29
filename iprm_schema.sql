-- ============================================================
-- Intelligent Process Resource Monitor (IPRM)
-- PostgreSQL Schema
-- ============================================================
-- Table groups:
--   A. Core telemetry      (1-3)
--   B. Analyzer output     (4-6)
--   C. ML / DL layer       (7-9)
--   D. Action + insight    (10-12)
-- ============================================================


-- ============================================================
-- A. CORE TELEMETRY
-- ============================================================

-- 1. Monitoring sessions -------------------------------------
-- One row each time the monitor agent starts. Lets you scope
-- queries to a run and record hardware context for the report.
CREATE TABLE monitoring_sessions (
    id                  BIGSERIAL PRIMARY KEY,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at            TIMESTAMPTZ,
    hostname            TEXT,
    os_name             TEXT,
    os_version          TEXT,
    cpu_model           TEXT,
    cpu_cores           INTEGER,
    cpu_threads         INTEGER,
    total_ram_mb        FLOAT,
    gpu_model           TEXT,
    total_gpu_memory_mb FLOAT,
    agent_version       TEXT
);


-- 2. System snapshots ----------------------------------------
-- The main time-series table. This is the LSTM's input source
-- and the backbone of every dashboard chart.
CREATE TABLE system_snapshots (
    id                      BIGSERIAL PRIMARY KEY,
    session_id              BIGINT REFERENCES monitoring_sessions(id),
    timestamp               TIMESTAMPTZ NOT NULL,

    -- CPU
    cpu_percent             FLOAT,
    cpu_freq_mhz            FLOAT,
    cpu_temp_c              FLOAT,
    load_avg_1m             FLOAT,          -- Linux/macOS; NULL on Windows

    -- Memory
    ram_percent             FLOAT,
    ram_used_mb             FLOAT,
    ram_available_mb        FLOAT,
    swap_percent            FLOAT,
    swap_used_mb            FLOAT,

    -- Disk
    disk_percent            FLOAT,
    disk_read_bytes         BIGINT,         -- cumulative counter
    disk_write_bytes        BIGINT,
    disk_read_rate_bps      FLOAT,          -- derived: delta / interval
    disk_write_rate_bps     FLOAT,

    -- GPU
    gpu_percent             FLOAT,
    gpu_memory_used_mb      FLOAT,
    gpu_memory_percent      FLOAT,
    gpu_temp_c              FLOAT,
    gpu_power_w             FLOAT,

    -- Network
    net_sent_bytes          BIGINT,
    net_recv_bytes          BIGINT,
    net_sent_rate_bps       FLOAT,
    net_recv_rate_bps       FLOAT,

    -- Context
    process_count           INTEGER,
    thread_count_total      INTEGER,
    boot_time               TIMESTAMPTZ,
    battery_percent         FLOAT,
    on_battery              BOOLEAN
);

CREATE INDEX idx_sys_snap_ts      ON system_snapshots (timestamp DESC);
CREATE INDEX idx_sys_snap_session ON system_snapshots (session_id, timestamp DESC);


-- 3. Process snapshots ---------------------------------------
-- Per-process detail for each system snapshot. Feeds the root
-- cause engine. ppid enables process-tree attribution.
CREATE TABLE process_snapshots (
    id                  BIGSERIAL PRIMARY KEY,
    snapshot_id         BIGINT NOT NULL REFERENCES system_snapshots(id) ON DELETE CASCADE,
    timestamp           TIMESTAMPTZ NOT NULL,

    pid                 INTEGER,
    ppid                INTEGER,
    name                TEXT,
    exe_path            TEXT,
    cmdline             TEXT,
    username            TEXT,
    status              TEXT,               -- running / sleeping / zombie

    cpu_percent         FLOAT,
    ram_percent         FLOAT,
    ram_rss_mb          FLOAT,
    ram_vms_mb          FLOAT,
    thread_count        INTEGER,
    open_files_count    INTEGER,
    io_read_bytes       BIGINT,
    io_write_bytes      BIGINT,
    gpu_percent         FLOAT,              -- NULL where per-process GPU is unavailable
    nice_value          INTEGER,
    create_time         TIMESTAMPTZ,

    is_dominant         BOOLEAN DEFAULT FALSE  -- flagged by the root cause engine
);

CREATE INDEX idx_proc_snap_snapshot ON process_snapshots (snapshot_id);
CREATE INDEX idx_proc_snap_name_ts  ON process_snapshots (name, timestamp DESC);
CREATE INDEX idx_proc_snap_pid_ts   ON process_snapshots (pid, timestamp DESC);


-- ============================================================
-- B. ANALYZER OUTPUT
-- ============================================================

-- 4. Chrome analyzer -----------------------------------------
-- Populated from the Chrome DevTools Protocol (/json endpoint).
CREATE TABLE chrome_tabs (
    id                  BIGSERIAL PRIMARY KEY,
    snapshot_id         BIGINT NOT NULL REFERENCES system_snapshots(id) ON DELETE CASCADE,
    timestamp           TIMESTAMPTZ NOT NULL,

    target_id           TEXT,               -- DevTools target id
    tab_title           TEXT,
    tab_url             TEXT,
    domain              TEXT,               -- parsed from url, easier to group by
    target_type         TEXT,               -- page / extension / service_worker / worker
    is_audible          BOOLEAN,
    is_active           BOOLEAN,
    is_media_playing    BOOLEAN,

    renderer_pid        INTEGER,            -- links back to process_snapshots.pid
    estimated_cpu       FLOAT,
    estimated_ram_mb    FLOAT
);

CREATE INDEX idx_chrome_tabs_snapshot ON chrome_tabs (snapshot_id);


-- 5. VS Code analyzer ----------------------------------------
-- One row per notable VS Code sub-process (extension host,
-- language server, git operation, terminal child).
CREATE TABLE vscode_activity (
    id                  BIGSERIAL PRIMARY KEY,
    snapshot_id         BIGINT NOT NULL REFERENCES system_snapshots(id) ON DELETE CASCADE,
    timestamp           TIMESTAMPTZ NOT NULL,

    pid                 INTEGER,
    component_type      TEXT,               -- extension_host / language_server /
                                            -- git / terminal / renderer / main
    component_name      TEXT,               -- 'tsserver', 'pylance', 'gitlens', ...
    workspace_path      TEXT,
    workspace_file_count INTEGER,
    cpu_percent         FLOAT,
    ram_mb              FLOAT,
    is_indexing         BOOLEAN,
    git_operation_active BOOLEAN
);

CREATE INDEX idx_vscode_snapshot ON vscode_activity (snapshot_id);


-- 6. Background task analyzer --------------------------------
-- Antivirus scans, cloud sync, OS updates, backup jobs.
CREATE TABLE background_task_activity (
    id                  BIGSERIAL PRIMARY KEY,
    snapshot_id         BIGINT NOT NULL REFERENCES system_snapshots(id) ON DELETE CASCADE,
    timestamp           TIMESTAMPTZ NOT NULL,

    pid                 INTEGER,
    task_category       TEXT,               -- antivirus / cloud_sync / os_update /
                                            -- backup / indexing
    task_name           TEXT,               -- 'Windows Defender', 'OneDrive', ...
    process_name        TEXT,
    cpu_percent         FLOAT,
    ram_mb              FLOAT,
    disk_read_rate_bps  FLOAT,
    disk_write_rate_bps FLOAT,
    is_scheduled        BOOLEAN,            -- scheduled job vs. user-triggered
    is_expected         BOOLEAN             -- suppresses false anomaly alerts
);

CREATE INDEX idx_bgtask_snapshot ON background_task_activity (snapshot_id);


-- ============================================================
-- C. ML / DL LAYER
-- ============================================================

-- 7. Model registry ------------------------------------------
-- Version + metrics for every trained Isolation Forest / LSTM.
CREATE TABLE model_registry (
    id                  BIGSERIAL PRIMARY KEY,
    model_type          TEXT NOT NULL,      -- 'isolation_forest' | 'lstm_forecaster'
    model_version       TEXT NOT NULL,
    trained_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    training_rows       INTEGER,
    training_start      TIMESTAMPTZ,        -- data window used
    training_end        TIMESTAMPTZ,
    feature_list        JSONB,              -- exact columns fed to the model
    hyperparameters     JSONB,              -- contamination, n_estimators, layers...
    scaler_params       JSONB,              -- min/max or mean/std for inference

    -- evaluation
    rmse                FLOAT,              -- LSTM
    mae                 FLOAT,              -- LSTM
    anomaly_rate        FLOAT,              -- Isolation Forest
    notes               TEXT,

    artifact_path       TEXT,               -- pickle / .h5 file location
    is_active           BOOLEAN DEFAULT FALSE
);

CREATE UNIQUE INDEX idx_model_active
    ON model_registry (model_type) WHERE is_active;


-- 8. Predictions ---------------------------------------------
-- LSTM forecasts. actual_value is backfilled once the horizon
-- elapses -> gives you prediction error, which is both an
-- accuracy metric and an Isolation Forest input feature.
CREATE TABLE predictions (
    id                      BIGSERIAL PRIMARY KEY,
    model_id                BIGINT REFERENCES model_registry(id),
    predicted_at            TIMESTAMPTZ NOT NULL,
    target_metric           TEXT NOT NULL,      -- 'cpu' | 'gpu' | 'ram'
    horizon_seconds         INTEGER NOT NULL,
    target_timestamp        TIMESTAMPTZ NOT NULL,   -- predicted_at + horizon

    predicted_value         FLOAT NOT NULL,
    confidence_lower        FLOAT,
    confidence_upper        FLOAT,

    actual_value            FLOAT,              -- backfilled
    prediction_error        FLOAT,              -- actual - predicted
    abs_percentage_error    FLOAT
);

CREATE INDEX idx_pred_target_ts ON predictions (target_metric, target_timestamp DESC);
CREATE INDEX idx_pred_pending   ON predictions (target_timestamp)
    WHERE actual_value IS NULL;


-- 9. Anomalies -----------------------------------------------
-- Isolation Forest detections.
CREATE TABLE anomalies (
    id                  BIGSERIAL PRIMARY KEY,
    snapshot_id         BIGINT NOT NULL REFERENCES system_snapshots(id) ON DELETE CASCADE,
    model_id            BIGINT REFERENCES model_registry(id),
    timestamp           TIMESTAMPTZ NOT NULL,

    anomaly_score       FLOAT NOT NULL,     -- raw Isolation Forest score
    is_anomaly          BOOLEAN NOT NULL,
    severity            TEXT,               -- low / medium / high / critical
    triggered_metrics   JSONB,              -- {"cpu": 92.1, "ram": 89.4}
    feature_vector      JSONB,              -- full input, for reproducibility

    duration_seconds    INTEGER,            -- filled when the anomaly ends
    resolved_at         TIMESTAMPTZ,
    was_suppressed      BOOLEAN DEFAULT FALSE,  -- e.g. expected gaming GPU load
    suppression_reason  TEXT
);

CREATE INDEX idx_anomalies_ts     ON anomalies (timestamp DESC);
CREATE INDEX idx_anomalies_active ON anomalies (is_anomaly, timestamp DESC)
    WHERE is_anomaly;


-- ============================================================
-- D. ACTION + INSIGHT
-- ============================================================

-- 10. Root causes --------------------------------------------
-- Output of the root cause engine for a given anomaly.
CREATE TABLE root_causes (
    id                      BIGSERIAL PRIMARY KEY,
    anomaly_id              BIGINT NOT NULL REFERENCES anomalies(id) ON DELETE CASCADE,
    timestamp               TIMESTAMPTZ NOT NULL,

    responsible_process     TEXT,           -- 'chrome.exe'
    responsible_pid         INTEGER,
    process_tree_pids       INTEGER[],      -- whole attributed subtree
    analyzer_used           TEXT,           -- 'chrome' | 'vscode' |
                                            -- 'background_task' | 'generic'
    analyzer_had_deep_data  BOOLEAN,        -- true = app API available,
                                            -- false = generic fallback

    explanation             TEXT,           -- human-readable narrative
    evidence                JSONB,          -- {"tab_count": 20, "media_tabs": 4}
    confidence              FLOAT,          -- 0-1
    explains_percent        FLOAT,          -- share of the load accounted for
    contributing_factors    JSONB           -- ranked secondary causes
);

CREATE INDEX idx_root_causes_anomaly ON root_causes (anomaly_id);


-- 11. Recommendations ----------------------------------------
-- Suggested fixes. Separate from root_causes so one cause can
-- yield several, and so you can track whether the user acted.
CREATE TABLE recommendations (
    id                  BIGSERIAL PRIMARY KEY,
    root_cause_id       BIGINT REFERENCES root_causes(id) ON DELETE CASCADE,
    timestamp           TIMESTAMPTZ NOT NULL,

    recommendation_text TEXT NOT NULL,
    action_category     TEXT,               -- close_tabs / enable_memory_saver /
                                            -- disable_extension / defer_scan
    priority            INTEGER,            -- 1 = highest
    estimated_impact    TEXT,               -- 'frees ~2.1 GB RAM'
    is_automatable      BOOLEAN,            -- can remediation execute it?

    shown_to_user       BOOLEAN DEFAULT FALSE,
    user_action         TEXT,               -- accepted / dismissed / ignored
    user_action_at      TIMESTAMPTZ
);


-- 12. Remediation actions ------------------------------------
-- Closed-loop interventions. The pre/post columns are the
-- measured evidence that an action changed system behaviour.
CREATE TABLE remediation_actions (
    id                      BIGSERIAL PRIMARY KEY,
    anomaly_id              BIGINT REFERENCES anomalies(id) ON DELETE CASCADE,
    root_cause_id           BIGINT REFERENCES root_causes(id),
    recommendation_id       BIGINT REFERENCES recommendations(id),
    timestamp               TIMESTAMPTZ NOT NULL,

    action_type             TEXT NOT NULL,  -- suspend_process / resume_process /
                                            -- lower_priority / set_affinity /
                                            -- trim_memory
    trigger_source          TEXT,           -- 'predictive' | 'reactive' | 'user'
    target_pid              INTEGER,
    target_process_name     TEXT,
    action_parameters       JSONB,          -- {"nice": 10} etc.

    -- measured effect
    pre_action_cpu          FLOAT,
    pre_action_ram_mb       FLOAT,
    post_action_cpu         FLOAT,          -- sampled N seconds later
    post_action_ram_mb      FLOAT,
    measurement_delay_sec   INTEGER,
    improvement_percent     FLOAT,

    success                 BOOLEAN,
    error_message           TEXT,
    reverted                BOOLEAN DEFAULT FALSE,
    reverted_at             TIMESTAMPTZ,
    revert_reason           TEXT            -- e.g. 'tab became active'
);

CREATE INDEX idx_remediation_ts ON remediation_actions (timestamp DESC);


-- 13. Health scores ------------------------------------------
-- Periodic composite score powering the dashboard gauge.
CREATE TABLE health_scores (
    id                  BIGSERIAL PRIMARY KEY,
    snapshot_id         BIGINT REFERENCES system_snapshots(id) ON DELETE CASCADE,
    timestamp           TIMESTAMPTZ NOT NULL,

    overall_score       FLOAT NOT NULL,     -- 0-100
    cpu_score           FLOAT,
    memory_score        FLOAT,
    disk_score          FLOAT,
    gpu_score           FLOAT,
    thermal_score       FLOAT,
    stability_score     FLOAT,              -- anomaly frequency component

    score_breakdown     JSONB,              -- weights + sub-scores, for the UI
    trend               TEXT                -- improving / stable / degrading
);

CREATE INDEX idx_health_ts ON health_scores (timestamp DESC);


-- ============================================================
-- RETENTION / ROLLUP
-- ============================================================
-- At 1 sample/sec you generate ~86k rows/day in system_snapshots
-- and far more in process_snapshots. Keep raw data for a short
-- window, roll the rest into this table, and delete the raw rows.

CREATE TABLE system_snapshots_hourly (
    id                  BIGSERIAL PRIMARY KEY,
    hour_bucket         TIMESTAMPTZ NOT NULL UNIQUE,
    sample_count        INTEGER,

    cpu_avg             FLOAT,
    cpu_max             FLOAT,
    cpu_p95             FLOAT,
    ram_avg             FLOAT,
    ram_max             FLOAT,
    gpu_avg             FLOAT,
    gpu_max             FLOAT,
    disk_read_total     BIGINT,
    disk_write_total    BIGINT,
    anomaly_count       INTEGER,
    avg_health_score    FLOAT
);

CREATE INDEX idx_hourly_bucket ON system_snapshots_hourly (hour_bucket DESC);


-- ============================================================
-- CONVENIENCE VIEW
-- ============================================================
-- Anomaly + cause + top recommendation, ready for the dashboard.

CREATE VIEW v_anomaly_report AS
SELECT
    a.id                AS anomaly_id,
    a.timestamp,
    a.severity,
    a.anomaly_score,
    a.triggered_metrics,
    rc.responsible_process,
    rc.analyzer_used,
    rc.explanation,
    rc.confidence,
    rc.explains_percent,
    r.recommendation_text,
    ra.action_type      AS remediation_taken,
    ra.improvement_percent
FROM anomalies a
LEFT JOIN root_causes rc          ON rc.anomaly_id = a.id
LEFT JOIN recommendations r       ON r.root_cause_id = rc.id AND r.priority = 1
LEFT JOIN remediation_actions ra  ON ra.anomaly_id = a.id
WHERE a.is_anomaly
ORDER BY a.timestamp DESC;
