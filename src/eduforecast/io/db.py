"""src/eduforecast/io/db.py"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """
    Context manager for SQLite connections.
    Ensures foreign keys are enabled and connection is closed properly.
    """
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        yield con
    finally:
        con.close()


def read_table(db_path: Path, table: str) -> pd.DataFrame:
    """Read an entire SQLite table into a DataFrame."""
    with connect(db_path) as con:
        return pd.read_sql(f"SELECT * FROM {table}", con)


def read_query(db_path: Path, query: str, params: Iterable | None = None) -> pd.DataFrame:
    """Read from SQLite using a SQL query."""
    with connect(db_path) as con:
        return pd.read_sql(query, con, params=params)


def write_table(
    db_path: Path,
    table: str,
    df: pd.DataFrame,
    *,
    if_exists: str = "replace",
    index: bool = False,
) -> None:
    """
    Write DataFrame to SQLite table.

    if_exists: 'replace' | 'append' | 'fail'
    """
    with connect(db_path) as con:
        df.to_sql(table, con, if_exists=if_exists, index=index)


def ensure_index(db_path: Path, table: str, cols: list[str], *, unique: bool = False) -> None:
    """Create an index on (cols) if it doesn't exist."""
    idx_name = f"idx_{table}_" + "_".join(cols)
    uniq = "UNIQUE" if unique else ""
    cols_sql = ", ".join(cols)

    sql = f"CREATE {uniq} INDEX IF NOT EXISTS {idx_name} ON {table} ({cols_sql});"
    with connect(db_path) as con:
        con.execute(sql)
        con.commit()
