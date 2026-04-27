import pandas as pd
import numpy as np
from config import (
    RAW_GAME_LOGS_PATH,
    RAW_INJURIES_PATH,
    CLEANED_TRAINING_DATA_PATH,
    FEATURES,
    TARGET,
    FEATURE_WINDOW_ACUTE,
    FEATURE_WINDOW_CHRONIC,
)


def clean_names(name):
    """Converts 'Last, First' to 'First Last' to match game logs."""
    if isinstance(name, str) and "," in name:
        parts = name.split(",")
        # parts[0] is Last, parts[1] is First. Strip removes extra spaces.
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name


def categorize_injury(reason):
    """Maps the raw injury text to our 3 ML classes."""
    reason = str(reason).lower()

    if pd.isna(reason) or reason == "nan":
        return 0  # Healthy

    # Class 2: Severe/Structural (High Severity Multiplier)
    if any(
        x in reason
        for x in ["tear", "fracture", "surgery", "rupture", "bone", "ligament"]
    ):
        return 2

    # Class 1: Soft Tissue/Wear-and-Tear (Moderate Severity Multiplier)
    elif any(
        x in reason
        for x in [
            "sprain",
            "strain",
            "spasm",
            "soreness",
            "tightness",
            "contusion",
            "cramp",
        ]
    ):
        return 1

    # Default fallback for illness, rest, or healthy
    return 0


def main():
    print("Loading raw CSVs...")
    games = pd.read_csv(RAW_GAME_LOGS_PATH)
    injuries = pd.read_csv(RAW_INJURIES_PATH)

    # --- 1. CLEAN DATES ---
    print("Standardizing dates...")
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    injuries["Game Date"] = pd.to_datetime(injuries["Game Date"])

    # --- 2. CLEAN NAMES ---
    print("Fixing name formats...")
    injuries["Player Name"] = injuries["Player Name"].apply(clean_names)

    # --- 3. CATEGORIZE INJURIES ---
    print("Mapping injury text to classes...")
    injuries["Injury_Class"] = injuries["Reason"].apply(categorize_injury)

    # We only care about the highest severity injury per player per day
    # (Sometimes players have two rows on the same day for multiple injuries)
    daily_injuries = (
        injuries.groupby(["Player Name", "Game Date"])["Injury_Class"]
        .max()
        .reset_index()
    )

    # --- 4. ENGINEER WORKLOAD FEATURES ---
    print("Calculating rolling workload features...")
    # Sort chronologically so rolling averages calculate correctly
    games = games.sort_values(by=["player_name", "GAME_DATE"])

    # Convert minutes to numeric (in case there are strings or NaNs)
    games["MIN"] = pd.to_numeric(games["MIN"], errors="coerce").fillna(0)

    # Calculate Acute Load (Average minutes over the last 5 games played)
    games["Acute_Load_5G"] = games.groupby("player_name")["MIN"].transform(
        lambda x: x.rolling(window=FEATURE_WINDOW_ACUTE, min_periods=1).mean()
    )

    # Calculate Chronic Load (Average minutes over the last 15 games played)
    games["Chronic_Load_15G"] = games.groupby("player_name")["MIN"].transform(
        lambda x: x.rolling(window=FEATURE_WINDOW_CHRONIC, min_periods=1).mean()
    )

    # --- 5. THE MERGE ---
    print("Merging datasets...")
    # We merge the injury report onto the game log.
    # Because injuries are reported the *day after* a game usually, we will merge the injury
    # to the game log that occurred 1 day prior, so the model learns that "Game X caused Injury Y".

    games["Next_Day"] = games["GAME_DATE"] + pd.Timedelta(days=1)

    merged_df = pd.merge(
        games,
        daily_injuries,
        left_on=["player_name", "Next_Day"],
        right_on=["Player Name", "Game Date"],
        how="left",
    )

    # Fill NaNs in the Injury_Class with 0 (Healthy)
    merged_df["Injury_Class"] = merged_df["Injury_Class"].fillna(0).astype(int)

    # --- 6. CLEAN UP AND EXPORT ---
    # Keep only the columns the ML model actually needs to learn from
    final_cols = [
        'player_name', 'GAME_DATE', 'MIN', 'Acute_Load_5G', 'Chronic_Load_15G', 'Injury_Class'
    ]
    ml_ready_df = merged_df[final_cols].copy()

    ml_ready_df.to_csv(CLEANED_TRAINING_DATA_PATH, index=False)
    print(f"Success! ML dataset created with shape: {ml_ready_df.shape}")


if __name__ == "__main__":
    main()
