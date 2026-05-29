"""DuckDB warehouse access: schema init, table loads, query helper."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

from nhs_forecast.config import Settings
from nhs_forecast.logging_setup import get_logger

log = get_logger("storage.warehouse")

SCHEMA_SQL = Path(__file__).with_name("schema.sql")

# curated lake dataset name -> warehouse fact/dim table
LOAD_MAP = {
    "procedures": "fact_procedures",
    "imaging": "fact_imaging",
    "rtt": "fact_rtt",
    "activity": "fact_activity",
    "demographics": "dim_demographics",
    "supply": "fact_supply",
}


@contextmanager
def connect(settings: Settings):
    settings.warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(settings.warehouse_path))
    try:
        yield con
    finally:
        con.close()


def init_schema(settings: Settings) -> None:
    with connect(settings) as con:
        con.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
    log.info("warehouse schema initialised at %s", settings.warehouse_path)


def replace_table(settings: Settings, table: str, df: pd.DataFrame) -> None:
    """Idempotent full-refresh load (truncate + insert) for a snapshot."""
    with connect(settings) as con:
        con.register("incoming", df)
        con.execute(f"DELETE FROM {table}")
        cols = ", ".join(con.execute(f"SELECT * FROM {table} LIMIT 0").df().columns)
        common = [c for c in df.columns if c in cols.split(", ")]
        col_list = ", ".join(common)
        con.execute(f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM incoming")
        con.unregister("incoming")
    log.info("loaded %d rows into %s", len(df), table)


def build_dimensions(settings: Settings) -> None:
    """Populate conformed dimensions from the loaded facts."""
    with connect(settings) as con:
        # dim_date from the union of fact dates
        con.execute(
            """
            DELETE FROM dim_date;
            INSERT INTO dim_date
            SELECT DISTINCT
                d AS date,
                year(d) AS year,
                quarter(d) AS quarter,
                month(d) AS month,
                strftime(d, '%B') AS month_name,
                CASE WHEN month(d) >= 4
                     THEN year(d) || '/' || ((year(d)+1) % 100)
                     ELSE (year(d)-1) || '/' || (year(d) % 100) END AS fiscal_year
            FROM (
                SELECT date AS d FROM fact_procedures
                UNION SELECT date FROM fact_imaging
                UNION SELECT date FROM fact_rtt
                UNION SELECT date FROM fact_activity
            ) WHERE d IS NOT NULL;
            """
        )
        con.execute(
            """
            DELETE FROM dim_trust;
            INSERT INTO dim_trust
            SELECT DISTINCT trust_code, region FROM fact_procedures WHERE trust_code IS NOT NULL;
            """
        )
    log.info("conformed dimensions built")


def query(settings: Settings, sql: str) -> pd.DataFrame:
    with connect(settings) as con:
        return con.execute(sql).df()
