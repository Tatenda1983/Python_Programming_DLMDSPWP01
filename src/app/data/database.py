import sqlite3
from typing import Optional

import pandas as pd


class DatabaseManager:
    """Wraps a sqlite3 connection and provides database helpers."""

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        c = self.conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                x REAL,
                y1 REAL,
                y2 REAL,
                y3 REAL,
                y4 REAL
            )
        """)

        ideal_cols = ", ".join(
            f"y{i} REAL" for i in range(1, 51)
        )

        c.execute(f"""
            CREATE TABLE IF NOT EXISTS ideal_functions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                x REAL,
                {ideal_cols}
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                x REAL,
                y REAL,
                delta_y REAL,
                ideal_func_no TEXT
            )
        """)

        self.conn.commit()

    def insert_dataframe(self, df: pd.DataFrame, table: str):
        """Insert a DataFrame into the named table."""

        if self.conn is None:
            raise RuntimeError("Database is not connected.")

        c = self.conn.cursor()
        c.execute(f"DELETE FROM {table}")

        df.to_sql(
            table,
            self.conn,
            if_exists="append",
            index=False
        )

        self.conn.commit()
        print(f"  {len(df):>4} rows -> table '{table}'")

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return the results as a DataFrame."""

        if self.conn is None:
            raise RuntimeError("Database is not connected.")

        return pd.read_sql_query(sql, self.conn)

    def close(self):
        """Close the database connection."""

        if self.conn:
            self.conn.close()
            self.conn = None