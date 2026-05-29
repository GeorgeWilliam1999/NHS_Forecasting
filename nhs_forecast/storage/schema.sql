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
