from app.config import (
    DB_PATH,
    IDEAL_CSV,
    TEST_CSV,
    TRAIN_CSV,
    VIZ_PATH,
)

from app.core.ideal_function_selector import IdealFunctionSelector
from app.core.test_point_mapper import TestPointMapper

from app.data.database import DatabaseManager
from app.data.training_data_handler import (
    IdealFunctionHandler,
    TestDataHandler,
    TrainingDataHandler,
)

from app.visualisation.visualiser import Visualiser


def main():
    separator = "=" * 60

    # ---------------------------------------------------------
    # STEP 1 - Load data
    # ---------------------------------------------------------

    print(f"\n{separator}\nSTEP 1 - Load data\n{separator}")

    train_handler = TrainingDataHandler(TRAIN_CSV)
    ideal_handler = IdealFunctionHandler(IDEAL_CSV)
    test_handler = TestDataHandler(TEST_CSV)

    train_df = train_handler.load()
    train_handler.validate()

    ideal_df = ideal_handler.load()
    ideal_handler.validate()

    test_df = test_handler.load()
    test_handler.validate()

    print(
        f"  Training : "
        f"{train_df.shape[0]} rows x {train_df.shape[1]} cols"
    )

    print(
        f"  Ideal    : "
        f"{ideal_df.shape[0]} rows x {ideal_df.shape[1]} cols"
    )

    print(
        f"  Test     : "
        f"{test_df.shape[0]} rows x {test_df.shape[1]} cols"
    )

    # ---------------------------------------------------------
    # STEP 2 - Persist data to SQLite
    # ---------------------------------------------------------

    print(
        f"\n{separator}\n"
        f"STEP 2 - Persist to SQLite\n"
        f"{separator}"
    )

    db = DatabaseManager(DB_PATH)
    db.connect()

    db.insert_dataframe(
        train_df,
        "training_data"
    )

    db.insert_dataframe(
        ideal_df,
        "ideal_functions"
    )

    # ---------------------------------------------------------
    # STEP 3 - Select ideal functions
    # ---------------------------------------------------------

    print(
        f"\n{separator}\n"
        f"STEP 3 - Select 4 best-fit ideal functions\n"
        f"{separator}"
    )

    selector = IdealFunctionSelector(
        train_df,
        ideal_df
    )

    chosen = selector.select()

    # ---------------------------------------------------------
    # STEP 4 - Map test data
    # ---------------------------------------------------------

    print(
        f"\n{separator}\n"
        f"STEP 4 - Map test data\n"
        f"{separator}"
    )

    mapper = TestPointMapper(
        ideal_df,
        chosen,
        selector.max_dev
    )

    results_df = mapper.map_all(test_df)

    matched = results_df["ideal_func_no"].notna().sum()
    unmatched = len(results_df) - matched

    print(f"  Matched   : {matched} / {len(results_df)}")
    print(f"  Unmatched : {unmatched}")
    print()
    print(results_df.to_string(index=False))

    # ---------------------------------------------------------
    # STEP 5 - Save test results
    # ---------------------------------------------------------

    print(
        f"\n{separator}\n"
        f"STEP 5 - Save test results to SQLite\n"
        f"{separator}"
    )

    db.insert_dataframe(
        results_df,
        "test_results"
    )

    db.close()

    # ---------------------------------------------------------
    # STEP 6 - Visualisation
    # ---------------------------------------------------------

    print(
        f"\n{separator}\n"
        f"STEP 6 - Visualise\n"
        f"{separator}"
    )

    visualiser = Visualiser(
        train_df,
        ideal_df,
        results_df,
        chosen
    )

    visualiser.plot(VIZ_PATH)

    print("\nApplication completed successfully.")
    print(f"  Database      : {DB_PATH}")
    print(f"  Visualisation : {VIZ_PATH}")


if __name__ == "__main__":
    main()