import pandas as pd


class IdealFunctionSelector:
    """
    Finds the ideal function that best fits each of the
    four training functions using least-squares error.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        ideal_df: pd.DataFrame
    ):
        self.train = train_df.set_index("x")
        self.ideal = ideal_df.set_index("x")

        self.chosen: dict[str, str] = {}
        self.max_dev: dict[str, float] = {}

    def select(self) -> dict[str, str]:
        """
        Select the best ideal function for each training function.

        Returns:
            Dictionary mapping each training column to its
            selected ideal-function column.
        """

        train_cols = list(self.train.columns)
        ideal_cols = list(self.ideal.columns)

        for train_col in train_cols:
            best_ideal = None
            best_sse = float("inf")

            for ideal_col in ideal_cols:
                residuals = (
                    self.train[train_col] -
                    self.ideal[ideal_col]
                )

                sse = float((residuals ** 2).sum())

                if sse < best_sse:
                    best_sse = sse
                    best_ideal = ideal_col

            self.chosen[train_col] = best_ideal

            self.max_dev[train_col] = float(
                (
                    self.train[train_col] -
                    self.ideal[best_ideal]
                ).abs().max()
            )

            print(
                f"  Train {train_col} -> "
                f"Ideal {best_ideal}  "
                f"SSE={best_sse:10.2f}  "
                f"max_dev={self.max_dev[train_col]:.4f}"
            )

        return self.chosen