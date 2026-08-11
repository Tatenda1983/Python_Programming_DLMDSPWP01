import unittest

import numpy as np
import pandas as pd

from app.core.ideal_function_selector import IdealFunctionSelector
from app.core.test_point_mapper import TestPointMapper
from app.data.data_handler import DataHandler
from app.data.training_data_handler import TrainingDataHandler
from app.exceptions import DataLoadError


class TestIdealFunctionSelector(unittest.TestCase):

    def _make_data(self):
        xs = np.array([0.0, 1.0, 2.0, 3.0])

        train = pd.DataFrame({
            "x": xs,
            "y1": xs,
            "y2": xs ** 2,
            "y3": np.sin(xs),
            "y4": np.cos(xs)
        })

        ideal = pd.DataFrame({
            "x": xs,
            **{
                f"y{i}": xs * i
                for i in range(1, 51)
            }
        })

        ideal["y1"] = xs
        ideal["y2"] = xs ** 2

        return train, ideal

    def test_selector_picks_correct_ideal(self):
        train, ideal = self._make_data()

        selector = IdealFunctionSelector(
            train,
            ideal
        )

        chosen = selector.select()

        self.assertEqual(
            chosen["y1"],
            "y1"
        )

        self.assertEqual(
            chosen["y2"],
            "y2"
        )

    def test_max_deviations_non_negative(self):
        train, ideal = self._make_data()

        selector = IdealFunctionSelector(
            train,
            ideal
        )

        selector.select()

        for value in selector.max_dev.values():
            self.assertGreaterEqual(value, 0)

    def test_four_choices_returned(self):
        train, ideal = self._make_data()

        selector = IdealFunctionSelector(
            train,
            ideal
        )

        chosen = selector.select()

        self.assertEqual(
            len(chosen),
            4
        )


class TestMapper(unittest.TestCase):

    def _make_mapper(self):
        xs = np.array([0.0, 1.0, 2.0])

        ideal = pd.DataFrame({
            "x": xs,
            **{
                f"y{i}": xs
                for i in range(1, 51)
            }
        })

        chosen = {
            "y1": "y1"
        }

        max_dev = {
            "y1": 0.5
        }

        return TestPointMapper(
            ideal,
            chosen,
            max_dev
        )

    def test_exact_match(self):
        mapper = self._make_mapper()

        ideal_func, delta = mapper.map_point(
            1.0,
            1.0
        )

        self.assertEqual(
            ideal_func,
            "y1"
        )

        self.assertAlmostEqual(
            delta,
            0.0
        )

    def test_within_threshold(self):
        mapper = self._make_mapper()

        ideal_func, delta = mapper.map_point(
            1.0,
            1.5
        )

        self.assertEqual(
            ideal_func,
            "y1"
        )

    def test_exceeds_threshold(self):
        mapper = self._make_mapper()

        ideal_func, delta = mapper.map_point(
            1.0,
            101.0
        )

        self.assertIsNone(
            ideal_func
        )

    def test_unknown_x_returns_none(self):
        mapper = self._make_mapper()

        ideal_func, delta = mapper.map_point(
            999.0,
            1.0
        )

        self.assertIsNone(
            ideal_func
        )


class TestDataHandlerClass(unittest.TestCase):

    def test_missing_file_raises(self):
        handler = DataHandler(
            "/no/such/file.csv"
        )

        with self.assertRaises(DataLoadError):
            handler.load()

    def test_training_validate_catches_missing_cols(self):
        handler = TrainingDataHandler.__new__(
            TrainingDataHandler
        )

        handler.df = pd.DataFrame({
            "x": [1],
            "y1": [1]
        })

        with self.assertRaises(DataLoadError):
            handler.validate()


if __name__ == "__main__":
    unittest.main()