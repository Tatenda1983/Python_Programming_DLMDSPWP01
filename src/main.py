"""
Programming With Python_DLMDSPWP01_Ideal _Function _Selection and Mapping Project Assignment
===============================================================
Method:
  - SQLite via stdlib sqlite3  
  - Visualisation using Bokeh 
  - Pandas and NumPy for computation
  - Full Object-Oriented Program design including inheritance and custom exceptions
  - Detailed unit tests (unittest)

"""

import sqlite3
import math
import os
import unittest
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from bokeh.plotting import figure, save
from bokeh.io import output_file
from bokeh.layouts import column, row, gridplot
from bokeh.models import Div, ColumnDataSource, Legend, LegendItem
from bokeh.palettes import Category10

# ─────────────────────────────────────────────────────────────────
# config.py
# ─────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
IDEAL_CSV = os.path.join(BASE_DIR, "ideal.csv")
TEST_CSV  = os.path.join(BASE_DIR, "test.csv")
DB_PATH   = os.path.join(BASE_DIR, "assignment.db")
VIZ_PATH  = os.path.join(BASE_DIR, "output.html")   # Bokeh -> interactive HTML

SQRT2 = math.sqrt(2)


# ─────────────────────────────────────────────────────────────────
# exceptions.py
# ─────────────────────────────────────────────────────────────────

class DataLoadError(Exception):
    """Raised when a data file cannot be loaded or validated."""
    pass


class MappingError(Exception):
    """Raised when test-point mapping fails unexpectedly."""
    pass


# ─────────────────────────────────────────────────────────────────
# database.py
# ─────────────────────────────────────────────────────────────────

class DatabaseManager:
    """It wraps a sqlite3 connection; provides create/insert/query helpers."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        c = self.conn.cursor()
        # Training data table
        c.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                x REAL, y1 REAL, y2 REAL, y3 REAL, y4 REAL
            )""")
        # Ideal functions table (51 columns: x + y1..y50)
        ideal_cols = ", ".join(f"y{i} REAL" for i in range(1, 51))
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS ideal_functions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                x REAL, {ideal_cols}
            )""")
        # Test results table
        c.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                x REAL, y REAL,
                delta_y REAL,
                ideal_func_no TEXT
            )""")
        self.conn.commit()

    def insert_dataframe(self, df: pd.DataFrame, table: str):
        """Insert a DataFrame into the named table, replacing existing rows."""
        c = self.conn.cursor()
        c.execute(f"DELETE FROM {table}")
        df.to_sql(table, self.conn, if_exists="append", index=False)
        self.conn.commit()
        print(f"  ✓ {len(df):>4} rows → table '{table}'")

    def query(self, sql: str) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn)

    def close(self):
        if self.conn:
            self.conn.close()


# ─────────────────────────────────────────────────────────────────
# data_handler.py
# ─────────────────────────────────────────────────────────────────
class DataHandler:
    """Its a base class: loads a CSV file and optionally validates it."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None
        print(f"Looking for file: {self.filepath}")

        if not os.path.exists(self.filepath):
            raise DataLoadError(f"File not found: {self.filepath}")

    def load(self) -> pd.DataFrame:
        """Load CSV. Raises DataLoadError on any problem."""
        try:
            self.df = pd.read_csv(self.filepath)
        except Exception as exc:
            raise DataLoadError(f"Cannot parse {self.filepath}: {exc}")
        if self.df.empty:
            raise DataLoadError(f"Empty file: {self.filepath}")
        return self.df

    def validate(self):
        """It overrides in subclass to check column names etc."""
        pass


# ─────────────────────────────────────────────────────────────────
# training_data_handler.py
# ─────────────────────────────────────────────────────────────────

class TrainingDataHandler(DataHandler):
    """It handles training CSV: columns x, y1, y2, y3, y4."""

    def validate(self):
        required = {"x", "y1", "y2", "y3", "y4"}
        missing  = required - set(self.df.columns)
        if missing:
            raise DataLoadError(f"Training data missing: {missing}")


class IdealFunctionHandler(DataHandler):
    """It handles ideal-functions CSV: columns x, y1 … y50."""

    def validate(self):
        required = {"x"} | {f"y{i}" for i in range(1, 51)}
        missing  = required - set(self.df.columns)
        if missing:
            raise DataLoadError(f"Ideal data missing {len(missing)} columns")


class TestDataHandler(DataHandler):
    """It handles test CSV: columns x, y."""

    def validate(self):
        required = {"x", "y"}
        missing  = required - set(self.df.columns)
        if missing:
            raise DataLoadError(f"Test data missing: {missing}")


# ─────────────────────────────────────────────────────────────────
# ideal_function_selector.py
# ─────────────────────────────────────────────────────────────────

class IdealFunctionSelector:
    """
    For each of the 4 training functions (y1–y4) finds the ideal function
    (y1–y50) that minimises the sum of squared deviations (Least-Square).
    """

    def __init__(self, train_df: pd.DataFrame, ideal_df: pd.DataFrame):
        self.train  = train_df.set_index("x")
        self.ideal  = ideal_df.set_index("x")
        self.chosen: dict[str, str]  = {}   # {train_col → ideal_col}
        self.max_dev: dict[str, float] = {}  # max |residual| per training col

    def select(self) -> dict:
        """
        Run selection. Populates self.chosen and self.max_dev.
        Returns self.chosen.
        """
        train_cols = list(self.train.columns)   # y1, y2, y3, y4
        ideal_cols = list(self.ideal.columns)   # y1 … y50

        for tc in train_cols:
            best_ic, best_sse = None, float("inf")
            for ic in ideal_cols:
                residuals = self.train[tc] - self.ideal[ic]
                sse = float((residuals ** 2).sum())
                if sse < best_sse:
                    best_sse, best_ic = sse, ic

            self.chosen[tc]  = best_ic
            self.max_dev[tc] = float((self.train[tc] - self.ideal[best_ic]).abs().max())
            print(f"  Train {tc} → Ideal {best_ic}  "
                  f"SSE={best_sse:10.2f}  max_dev={self.max_dev[tc]:.4f}")

        return self.chosen


# ─────────────────────────────────────────────────────────────────
# test_point_mapper.py
# ─────────────────────────────────────────────────────────────────

class TestPointMapper:
    """
    It maps each test (x, y) to one of the 4 chosen ideal functions.

    Criterion (from spec):
        |y_test – y_ideal(x)| ≤ max_train_deviation(chosen ideal) × √2
    """

    def __init__(self, ideal_df: pd.DataFrame,
                 chosen: dict, max_dev: dict):
        self.ideal   = ideal_df.set_index("x")
        self.chosen  = chosen   # {train_col: ideal_col}
        self.max_dev = max_dev

    def map_point(self, x: float, y: float) -> tuple[Optional[str], Optional[float]]:
        """
        It returns (ideal_col, delta) for the best qualifying ideal function,
        or (None, None) if no function meets the criterion.
        """
        best_ic, best_delta = None, float("inf")

        for tc, ic in self.chosen.items():
            if x not in self.ideal.index:
                continue
            threshold = self.max_dev[tc] * SQRT2
            y_ideal   = float(self.ideal.at[x, ic])
            delta     = abs(y - y_ideal)
            if delta <= threshold and delta < best_delta:
                best_delta, best_ic = delta, ic

        if best_ic is None:
            return None, None
        return best_ic, round(best_delta, 6)

    def map_all(self, test_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row_ in test_df.iterrows():
            ic, delta = self.map_point(row_["x"], row_["y"])
            rows.append({"x": row_["x"], "y": row_["y"],
                         "delta_y": delta, "ideal_func_no": ic})
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# visualiser.py  
# ─────────────────────────────────────────────────────────────────

class Visualiser:
    """Generation of a multi-panel Bokeh layout (The file is saved as an HTML file)."""

    COLOURS = Category10[10]  # Bokeh's built-in categorical palette

    def __init__(self, train_df, ideal_df, results_df, chosen):
        self.train   = train_df
        self.ideal   = ideal_df
        self.results = results_df
        self.chosen  = chosen

    def _train_vs_ideal_panels(self):
        """4 small figures, one per training/ideal pair (replaces the 2x2 gridspec)."""
        panels = []
        for tc, ic in self.chosen.items():
            p = figure(width=380, height=280,
                       title=f"Train {tc}  →  Ideal {ic}",
                       tools="pan,wheel_zoom,box_zoom,reset,save")
            p.line(self.train["x"], self.train[tc],
                   line_color=self.COLOURS[0], line_width=2,
                   legend_label=f"Train {tc}")
            p.line(self.ideal["x"], self.ideal[ic],
                   line_color=self.COLOURS[1], line_width=1.5,
                   line_dash="dashed", legend_label=f"Ideal {ic}")
            p.legend.label_text_font_size = "7pt"
            p.legend.location = "top_left"
            p.legend.background_fill_alpha = 0.6
            panels.append(p)
        return panels

    def _summary_div(self):
        
        matched = self.results["ideal_func_no"].notna().sum()
        lines = ["<b>Chosen Ideal Functions</b>", "<hr style='margin:4px 0'>"]
        for tc, ic in self.chosen.items():
            lines.append(f"{tc} &rarr; {ic}")
        lines.append("<br>")
        lines.append(f"Test points : {len(self.results)}")
        lines.append(f"Matched : {matched}")
        lines.append(f"Unmatched : {len(self.results) - matched}")
        html = (
            "<div style='font-family:monospace; font-size:13px; "
            "background:#f0f0f0; border:1px solid #aaa; border-radius:6px; "
            "padding:12px; width:340px;'>" + "<br>".join(lines) + "</div>"
        )
        return Div(text=html, width=340, height=280)

    def _test_mapping_plot(self):
        """Wide bottom figure: test points coloured by matched ideal function."""
        p = figure(width=1180, height=420,
                   title="Test Data Mapping to Chosen Ideal Functions",
                   x_axis_label="x", y_axis_label="y",
                   tools="pan,wheel_zoom,box_zoom,reset,save,hover",
                   tooltips=[("x", "@x"), ("y", "@y"), ("delta", "@delta_y"),
                             ("ideal fn", "@ideal_func_no")])

        ideal_cols = list(self.chosen.values())
        col_map = {ic: self.COLOURS[j % len(self.COLOURS)]
                   for j, ic in enumerate(ideal_cols)}

        for ic in ideal_cols:
            # faint ideal curve
            p.line(self.ideal["x"], self.ideal[ic],
                   line_color=col_map[ic], line_width=1.2,
                   line_dash="dashed", line_alpha=0.4)

            sub = self.results[self.results["ideal_func_no"] == ic]
            src = ColumnDataSource(sub)
            p.scatter("x", "y", source=src, size=8,
                      color=col_map[ic], legend_label=f"→ {ic}")

        unmatched = self.results[self.results["ideal_func_no"].isna()]
        if not unmatched.empty:
            src_un = ColumnDataSource(unmatched)
            p.scatter("x", "y", source=src_un, size=8, marker="x",
                      color="grey", legend_label="Unmatched")

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"   # click legend entries to toggle series
        return p

    def plot(self, save_path: str):
        output_file(save_path, title="Python Assignment — Results")

        top_grid = gridplot(self._train_vs_ideal_panels(), ncols=2)
        top_row = row(top_grid, self._summary_div())
        bottom = self._test_mapping_plot()

        title_div = Div(text="<h2>Python Assignment — Results</h2>")
        layout = column(title_div, top_row, bottom)

        save(layout)
        print(f"  ✓ Visualisation saved → {save_path}")


# ─────────────────────────────────────────────────────────────────
# test_ideal_function_selector.py
# ─────────────────────────────────────────────────────────────────

class TestIdealFunctionSelector(unittest.TestCase):

    def _make_data(self):
        xs = np.array([0.0, 1.0, 2.0, 3.0])
        train = pd.DataFrame({"x": xs, "y1": xs, "y2": xs**2,
                               "y3": np.sin(xs), "y4": np.cos(xs)})
        ideal = pd.DataFrame({"x": xs,
                               **{f"y{i}": xs * i for i in range(1, 51)}})
        ideal["y1"] = xs        # perfect match for train y1
        ideal["y2"] = xs ** 2  # perfect match for train y2
        return train, ideal

    def test_selector_picks_correct_ideal(self):
        train, ideal = self._make_data()
        sel = IdealFunctionSelector(train, ideal)
        chosen = sel.select()
        self.assertEqual(chosen["y1"], "y1")
        self.assertEqual(chosen["y2"], "y2")

    def test_max_deviations_non_negative(self):
        train, ideal = self._make_data()
        sel = IdealFunctionSelector(train, ideal)
        sel.select()
        for v in sel.max_dev.values():
            self.assertGreaterEqual(v, 0)

    def test_four_choices_returned(self):
        train, ideal = self._make_data()
        sel = IdealFunctionSelector(train, ideal)
        chosen = sel.select()
        self.assertEqual(len(chosen), 4)


class TestMapper(unittest.TestCase):

    def _make_mapper(self):
        xs = np.array([0.0, 1.0, 2.0])
        ideal = pd.DataFrame({"x": xs, **{f"y{i}": xs for i in range(1, 51)}})
        chosen  = {"y1": "y1"}
        max_dev = {"y1": 0.5}
        return TestPointMapper(ideal, chosen, max_dev)

    def test_exact_match(self):
        m = self._make_mapper()
        ic, delta = m.map_point(1.0, 1.0)
        self.assertEqual(ic, "y1")
        self.assertAlmostEqual(delta, 0.0)

    def test_within_threshold(self):
        m = self._make_mapper()
        # threshold = 0.5 * sqrt(2) ≈ 0.707; delta = 0.5 should pass
        ic, delta = m.map_point(1.0, 1.5)
        self.assertEqual(ic, "y1")

    def test_exceeds_threshold(self):
        m = self._make_mapper()
        # threshold ≈ 0.707; delta = 100 should fail
        ic, delta = m.map_point(1.0, 101.0)
        self.assertIsNone(ic)

    def test_unknown_x_returns_none(self):
        m = self._make_mapper()
        ic, delta = m.map_point(999.0, 1.0)
        self.assertIsNone(ic)


class TestDataHandlerClass(unittest.TestCase):

    def test_missing_file_raises(self):
        h = DataHandler("/no/such/file.csv")
        with self.assertRaises(DataLoadError):
            h.load()

    def test_training_validate_catches_missing_cols(self):
        h = TrainingDataHandler.__new__(TrainingDataHandler)
        h.df = pd.DataFrame({"x": [1], "y1": [1]})   # missing y2–y4
        with self.assertRaises(DataLoadError):
            h.validate()


# ─────────────────────────────────────────────────────────────────
# main.py
# ─────────────────────────────────────────────────────────────────
def main():
    sep = "=" * 60

    # Stage 1: Data Load
    print(f"\n{sep}\nSTEP 1 — Load data\n{sep}")

    train_h = TrainingDataHandler(TRAIN_CSV)
    ideal_h = IdealFunctionHandler(IDEAL_CSV)
    test_h  = TestDataHandler(TEST_CSV)

    train_df = train_h.load()
    train_h.validate()

    ideal_df = ideal_h.load()
    ideal_h.validate()

    test_df = test_h.load()
    test_h.validate()

    print(f"  Training : {train_df.shape[0]} rows × {train_df.shape[1]} cols")
    print(f"  Ideal    : {ideal_df.shape[0]} rows × {ideal_df.shape[1]} cols")
    print(f"  Test     : {test_df.shape[0]} rows × {test_df.shape[1]} cols")

    # Stage 2: Database Paths
    print(f"\n{sep}\nSTEP 2 — Persist to SQLite\n{sep}")
    db = DatabaseManager(DB_PATH)
    db.connect()
    db.insert_dataframe(train_df, "training_data")
    db.insert_dataframe(ideal_df, "ideal_functions")

    # Stage 3: Ideal function selection
    print(f"\n{sep}\nSTEP 3 — Select 4 best-fit ideal functions\n{sep}")
    selector = IdealFunctionSelector(train_df, ideal_df)
    chosen   = selector.select()

    # Stage 4: Test data mapped
    print(f"\n{sep}\nSTEP 4 — Map test data\n{sep}")
    mapper     = TestPointMapper(ideal_df, chosen, selector.max_dev)
    results_df = mapper.map_all(test_df)

    matched   = results_df["ideal_func_no"].notna().sum()
    unmatched = len(results_df) - matched
    print(f"  Matched   : {matched} / {len(results_df)}")
    print(f"  Unmatched : {unmatched}")
    print()
    print(results_df.to_string(index=False))

    # Stage 5: Results saved
    print(f"\n{sep}\nSTEP 5 — Save test results to SQLite\n{sep}")
    db.insert_dataframe(results_df, "test_results")
    db.close()

    # Stage 6: Visualisation
    print(f"\n{sep}\nSTEP 6 — Visualise\n{sep}")
    viz = Visualiser(train_df, ideal_df, results_df, chosen)
    viz.plot(VIZ_PATH)

    # Stage 7: Unit tests
    print(f"\n{sep}\nSTEP 7 — Unit tests\n{sep}")
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in (TestIdealFunctionSelector, TestMapper, TestDataHandlerClass):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    status = "SUCCEDED" if result.wasSuccessful() else "FAILED"
    print(f"\nCompleted — {status}")
    print(f"   Database      : {DB_PATH}")
    print(f"   Visualisation : {VIZ_PATH}")


if __name__ == "__main__":
    main()
