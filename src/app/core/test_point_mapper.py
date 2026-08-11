from typing import Optional

import pandas as pd

from app.config import SQRT2


class TestPointMapper:
    """
    Maps each test point to one of the four selected ideal functions.
    """

    def __init__(
        self,
        ideal_df: pd.DataFrame,
        chosen: dict[str, str],
        max_dev: dict[str, float]
    ):
        self.ideal = ideal_df.set_index("x")
        self.chosen = chosen
        self.max_dev = max_dev

    def map_point(
        self,
        x: float,
        y: float
    ) -> tuple[Optional[str], Optional[float]]:
        """
        Return the best qualifying ideal function and its delta.

        A point qualifies when:

            |y_test - y_ideal| <= max_deviation * sqrt(2)

        Returns:
            (ideal_function, delta)

        If no ideal function qualifies:
            (None, None)
        """

        best_ideal = None
        best_delta = float("inf")

        for train_col, ideal_col in self.chosen.items():

            if x not in self.ideal.index:
                continue

            threshold = self.max_dev[train_col] * SQRT2

            y_ideal = float(
                self.ideal.at[x, ideal_col]
            )

            delta = abs(y - y_ideal)

            if delta <= threshold and delta < best_delta:
                best_delta = delta
                best_ideal = ideal_col

        if best_ideal is None:
            return None, None

        return best_ideal, round(best_delta, 6)

    def map_all(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Map all test points and return the results as a DataFrame."""

        rows = []

        for _, row in test_df.iterrows():

            ideal_func, delta = self.map_point(
                row["x"],
                row["y"]
            )

            rows.append({
                "x": row["x"],
                "y": row["y"],
                "delta_y": delta,
                "ideal_func_no": ideal_func
            })

        return pd.DataFrame(rows)