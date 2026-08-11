import os
from typing import Optional

import pandas as pd

from app.exceptions import DataLoadError


class DataHandler:
    """Base class that loads a CSV file and optionally validates it."""

    def __init__(self, filepath: str):
        self.filepath = str(filepath)
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load CSV and raise DataLoadError if there is a problem."""

        if not os.path.exists(self.filepath):
            raise DataLoadError(
                f"File not found: {self.filepath}"
            )

        try:
            self.df = pd.read_csv(self.filepath)
        except Exception as exc:
            raise DataLoadError(
                f"Cannot parse {self.filepath}: {exc}"
            ) from exc

        if self.df.empty:
            raise DataLoadError(
                f"Empty file: {self.filepath}"
            )

        return self.df

    def validate(self):
        """Override in subclasses to check column names."""

        pass