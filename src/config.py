import os
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# ============================================
# DATA INGESTION
# ============================================
SAMPLE_PLAYER_COUNT = 20
SEASONS = ["2024-25", "2025-26"]
API_DELAY_SECONDS = 0.5
INJURY_REPORT_HOUR = 17
INJURY_REPORT_MINUTE = 30
INJURY_API_DELAY_SECONDS = 0.3
SEASON_START_DATE = datetime(2025, 10, 22)

# ============================================
# PATHS
# ============================================
RAW_GAME_LOGS_PATH = DATA_DIR / "raw_game_logs.csv"
RAW_INJURIES_PATH = DATA_DIR / "raw_injuries.csv"
CLEANED_TRAINING_DATA_PATH = DATA_DIR / "cleaned_training_data.csv"
MODEL_PATH = MODELS_DIR / "xgboost_injury_model.pkl"

# ============================================
# MODEL TRAINING
# ============================================
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_MAX_DEPTH = 4
MODEL_LEARNING_RATE = 0.1
MODEL_N_ESTIMATORS = 100

# ============================================
# FEATURES
# ============================================
FEATURES = ["MIN", "Acute_Load_5G", "Chronic_Load_15G"]
TARGET = "Injury_Class"
FEATURE_WINDOW_ACUTE = 5
FEATURE_WINDOW_CHRONIC = 15

# ============================================
# INJURY CLASSIFICATION
# ============================================
CLASS_LABELS = {0: "Healthy", 1: "Soft Tissue", 2: "Severe/Structural"}
SEVERITY_MULTIPLIERS = {0: 1.0, 1: 1.5, 2: 2.5}
