-- DuckDB warehouse schema for the NHS equipment forecasting system.
-- Star-ish design: conformed dimensions (date, trust, equipment) + fact tables.
-- DuckDB is used as a zero-ops analytical warehouse; the same DDL ports to
-- Postgres/BigQuery with minimal changes (types are ANSI-ish).

-- ---------- dimensions ----------
CREATE TABLE IF NOT EXISTS dim_date (
    date            DATE PRIMARY KEY,
    year            INTEGER,
    quarter         INTEGER,
    month           INTEGER,
    month_name      VARCHAR,
    fiscal_year     VARCHAR        -- NHS financial year, Apr-Mar
);

CREATE TABLE IF NOT EXISTS dim_trust (
    trust_code      VARCHAR PRIMARY KEY,
    region          VARCHAR
);

CREATE TABLE IF NOT EXISTS dim_equipment (
    equipment_type  VARCHAR PRIMARY KEY,
    category        VARCHAR,       -- imaging | surgical | implant | device | consumable
    unit_cost_gbp   DOUBLE,
    reusable        BOOLEAN        -- reusable capital asset vs single-use consumable
);

-- procedure_code -> equipment_type mapping with weights (see mapping module)
CREATE TABLE IF NOT EXISTS dim_proc_equipment_map (
    procedure_code  VARCHAR,
    equipment_type  VARCHAR,
    weight          DOUBLE,        -- expected units of equipment per procedure
    map_type        VARCHAR,       -- rule | probabilistic
    source          VARCHAR,
    PRIMARY KEY (procedure_code, equipment_type)
);

-- ---------- facts ----------
CREATE TABLE IF NOT EXISTS fact_procedures (
    date            DATE,
    trust_code      VARCHAR,
    region          VARCHAR,
    opcs_chapter    VARCHAR,
    procedure_code  VARCHAR,
    n_procedures    BIGINT
);

CREATE TABLE IF NOT EXISTS fact_imaging (
    date            DATE,
    trust_code      VARCHAR,
    region          VARCHAR,
    modality        VARCHAR,
    n_tests         BIGINT
);

CREATE TABLE IF NOT EXISTS fact_rtt (
    date                DATE,
    trust_code          VARCHAR,
    region              VARCHAR,
    treatment_function  VARCHAR,
    waiting_list_size   BIGINT,
    pct_within_18wk     DOUBLE
);

CREATE TABLE IF NOT EXISTS fact_activity (
    date            DATE,
    trust_code      VARCHAR,
    region          VARCHAR,
    activity_type   VARCHAR,
    n_activity      BIGINT
);

CREATE TABLE IF NOT EXISTS dim_demographics (
    year            INTEGER,
    region          VARCHAR,
    age_band        VARCHAR,
    population       BIGINT
);

CREATE TABLE IF NOT EXISTS fact_supply (
    quarter         VARCHAR,
    category        VARCHAR,
    spend_gbp       DOUBLE,
    units           BIGINT
);

-- ---------- outputs ----------
CREATE TABLE IF NOT EXISTS forecast_procedures (
    run_id          VARCHAR,
    model           VARCHAR,
    level           VARCHAR,       -- national | regional | trust
    trust_code      VARCHAR,
    region          VARCHAR,
    procedure_code  VARCHAR,
    date            DATE,
    yhat            DOUBLE,
    yhat_lower      DOUBLE,
    yhat_upper      DOUBLE
);

CREATE TABLE IF NOT EXISTS forecast_equipment (
    run_id          VARCHAR,
    scenario        VARCHAR,
    level           VARCHAR,
    trust_code      VARCHAR,
    region          VARCHAR,
    equipment_type  VARCHAR,
    date            DATE,
    demand          DOUBLE,
    demand_lower    DOUBLE,
    demand_upper    DOUBLE
);

-- ---------- pay-per-use telemetry / underwriting ----------
-- Device-session is the underwriting unit of observation (see telemetry module).
-- These tables are additive; the aggregate demand pipeline does not depend on them.
CREATE TABLE IF NOT EXISTS dim_device (
    device_id            VARCHAR PRIMARY KEY,
    device_type          VARCHAR,
    site_id              VARCHAR,
    region               VARCHAR,
    specialty            VARCHAR,
    install_date         DATE,
    contract_id          VARCHAR,
    price_per_session_gbp DOUBLE,
    monthly_fixed_cost_gbp DOUBLE,   -- capital charge + maintenance the platform carries
    min_monthly_floor_gbp DOUBLE,    -- contractual revenue floor (left-censoring)
    cap_sessions_day     INTEGER     -- contractual / physical daily cap (right-censoring)
);

CREATE TABLE IF NOT EXISTS dim_operator (
    operator_hash   VARCHAR PRIMARY KEY,
    site_id         VARCHAR,
    specialty       VARCHAR,
    role            VARCHAR
);

-- Raw, append-only telemetry event log (immutable source of truth).
CREATE TABLE IF NOT EXISTS telemetry_event (
    event_id        VARCHAR,
    device_id       VARCHAR,
    site_id         VARCHAR,
    operator_hash   VARCHAR,
    event_type      VARCHAR,   -- power_on|power_off|active_start|active_end|error|login
    ts_device       TIMESTAMP,
    seq_no          BIGINT,
    active_seconds  DOUBLE,
    n_errors        INTEGER,
    procedure_code  VARCHAR,
    ingest_quality  VARCHAR
);

-- Derived sessions (rebuildable from telemetry_event).
CREATE TABLE IF NOT EXISTS fact_device_session (
    session_id      VARCHAR,
    device_id       VARCHAR,
    site_id         VARCHAR,
    operator_hash   VARCHAR,
    procedure_code  VARCHAR,
    t_start         TIMESTAMP,
    t_end           TIMESTAMP,
    active_seconds  DOUBLE,
    n_errors        INTEGER,
    billable        BOOLEAN,
    billed_amount_gbp DOUBLE
);

-- Modelling grain: one row per device per day.
CREATE TABLE IF NOT EXISTS fact_device_day (
    device_id       VARCHAR,
    date            DATE,
    n_sessions      INTEGER,
    n_billable      INTEGER,
    active_seconds  DOUBLE,
    exposure_hours  DOUBLE,    -- powered-on hours (offset; 0 => downtime/censored)
    n_errors        INTEGER,
    n_operators     INTEGER,
    billed_gbp      DOUBLE
);

-- Per-device underwriting output.
CREATE TABLE IF NOT EXISTS underwriting_device (
    run_id          VARCHAR,
    device_id       VARCHAR,
    site_id         VARCHAR,
    region          VARCHAR,
    specialty       VARCHAR,
    expected_sessions DOUBLE,
    expected_revenue_gbp DOUBLE,
    cv              DOUBLE,     -- coefficient of variation of horizon revenue
    beta_book       DOUBLE,     -- covariance with the rest of the book
    suggested_price_gbp DOUBLE,
    current_price_gbp DOUBLE,
    uar_p5_sessions DOUBLE,     -- Utilisation-at-Risk (downside)
    floor_breach_prob DOUBLE,
    op_herfindahl   DOUBLE,     -- operator concentration (key-person risk)
    expected_margin_gbp DOUBLE
);
