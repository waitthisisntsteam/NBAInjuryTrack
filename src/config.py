import os
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ORIGINAL_DATA_DIR = DATA_DIR / "original"
PREP_DATA_DIR = DATA_DIR / "prep_data"
TRAINING_DATA_DIR = DATA_DIR / "training"
MODELS_DIR = BASE_DIR / "models"

# ============================================
# DATA INGESTION
# ============================================
SAMPLE_PLAYER_COUNT = 600
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
API_DELAY_SECONDS = 2
INJURY_REPORT_HOUR = 24
INJURY_REPORT_MINUTE = 0
INJURY_API_DELAY_SECONDS = 2
SEASON_START_DATE = datetime(2021, 10, 22)

# ============================================
# PATHS
# ============================================
KAGGLE_GAME_LOGS_PATH = ORIGINAL_DATA_DIR / "PlayerStatisticsExtended.csv"
KAGGLE_PLAYERS_PATH = ORIGINAL_DATA_DIR / "Players.csv"
KAGGLE_INJURIES_PATH = ORIGINAL_DATA_DIR / "injury_data.csv"
RAW_GAME_LOGS_PATH = PREP_DATA_DIR / "raw_game_logs.csv"
RAW_INJURIES_PATH = PREP_DATA_DIR / "raw_injuries.csv"
CLEANED_TRAINING_DATA_PATH = TRAINING_DATA_DIR / "cleaned_training_data.csv"
CLEANED_INJURIES_PATH = PREP_DATA_DIR / "cleaned_injuries.csv"
ACTIVE_PLAYERS_PATH = PREP_DATA_DIR / "active_players.csv"
MODEL_PATH = MODELS_DIR / "xgboost_injury_model.pkl"

# ============================================
# MODEL TRAINING
# ============================================
TEST_SIZE = 0.2
RANDOM_STATE = 5
MODEL_MAX_DEPTH = 4
MODEL_LEARNING_RATE = 0.01
MODEL_N_ESTIMATORS = 100

# ============================================
# FEATURES
# ============================================
FEATURES = [
    "MIN",
    "Hyper_Acute_Load_3G",
    "Acute_Load_5G",
    "Chronic_Load_15G",
    "ACWR",
    "Minutes_Last_7D",
    "Minutes_Last_14D",
    "Minutes_Last_30D",
    "Usage_Pace_Mix",
    "Position_Group",
    "Back_to_Back",
    "BMI",
    "BMI_Tier",
    "Height_Tier",
    "Weight_Tier",
]
TARGET = "Injury_Class"
FEATURE_WINDOW_ACUTE = 5
FEATURE_WINDOW_CHRONIC = 15

# ============================================
# INJURY CLASSIFICATION
# ============================================
CLASS_LABELS = {
    0: "Healthy",
    1: "Fatigue/Soreness",
    2: "Contusion/Impact",
    3: "Joint Sprain",
    4: "Muscular Strain",
    5: "Structural/Surgical",
}
SEVERITY_MULTIPLIERS = {
    0: 1.0,
    1: 1.2,
    2: 1.4,
    3: 2.0,
    4: 2.5,
    5: 4.0,
}
