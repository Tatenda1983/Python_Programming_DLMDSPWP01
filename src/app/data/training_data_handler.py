from app.data.data_handler import DataHandler
from app.exceptions import DataLoadError


class TrainingDataHandler(DataHandler):
    """Handles training CSV: columns x, y1, y2, y3, y4."""

    def validate(self):
        required = {"x", "y1", "y2", "y3", "y4"}
        missing = required - set(self.df.columns)

        if missing:
            raise DataLoadError(
                f"Training data missing: {missing}"
            )


class IdealFunctionHandler(DataHandler):
    """Handles ideal-functions CSV: columns x, y1 ... y50."""

    def validate(self):
        required = {"x"} | {f"y{i}" for i in range(1, 51)}
        missing = required - set(self.df.columns)

        if missing:
            raise DataLoadError(
                f"Ideal data missing {len(missing)} columns"
            )


class TestDataHandler(DataHandler):
    """Handles test CSV: columns x, y."""

    def validate(self):
        required = {"x", "y"}
        missing = required - set(self.df.columns)

        if missing:
            raise DataLoadError(
                f"Test data missing: {missing}"
            )