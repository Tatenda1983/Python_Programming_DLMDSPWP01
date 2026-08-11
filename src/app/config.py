from pathlib import Path
import math

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset directory
DATASET_DIR = PROJECT_ROOT / "datasets"

TRAIN_CSV = DATASET_DIR / "train.csv"
IDEAL_CSV = DATASET_DIR / "ideal.csv"
TEST_CSV = DATASET_DIR / "test.csv"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output"

# SQLite database
DB_PATH = OUTPUT_DIR / "assignment.db"

# Bokeh visualisation
VIZ_PATH = OUTPUT_DIR / "output.html"

# Mapping threshold
SQRT2 = math.sqrt(2)