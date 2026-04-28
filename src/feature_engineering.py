import pandas as pd

from config import (
    RAW_GAME_LOGS_PATH,
    CLEANED_INJURIES_PATH,
    CLEANED_TRAINING_DATA_PATH,
)


START_DATE = pd.Timestamp("2016-01-01")
END_DATE = pd.Timestamp("2025-12-31")


def categorize_injury(reason):
    reason = str(reason).lower()
    if pd.isna(reason) or reason == "nan":
        return 0
    if any(
        x in reason
        for x in [
            "tear",
            "rupture",
            "fracture",
            "surgery",
            "broke",
            "acl",
            "mcl",
            "achilles",
            "meniscus",
            "bone",
            "dislocation",
        ]
    ):
        return 5
    if any(x in reason for x in ["strain", "pull"]):
        return 4
    if any(x in reason for x in ["sprain", "hyperextension"]):
        return 3
    if any(x in reason for x in ["contusion", "bruise"]):
        return 2
    if any(x in reason for x in ["sore", "tight", "spasm", "cramp", "stiffness"]):
        return 1
    return 0


def add_workload_features(games):
    games = games.sort_values(by=["player_name", "GAME_DATE"])
    games["MIN"] = pd.to_numeric(games["MIN"], errors="coerce").fillna(0)

    games["Minutes_Last_7D"] = games.groupby("player_name")[["GAME_DATE", "MIN"]].apply(
        lambda x: x.rolling("7D", on="GAME_DATE")["MIN"].sum()
    ).reset_index(level=0, drop=True)
    games["Minutes_Last_14D"] = games.groupby("player_name")[["GAME_DATE", "MIN"]].apply(
        lambda x: x.rolling("14D", on="GAME_DATE")["MIN"].sum()
    ).reset_index(level=0, drop=True)
    games["Minutes_Last_30D"] = games.groupby("player_name")[["GAME_DATE", "MIN"]].apply(
        lambda x: x.rolling("30D", on="GAME_DATE")["MIN"].sum()
    ).reset_index(level=0, drop=True)

    games["Hyper_Acute_Load_3G"] = games.groupby("player_name")["MIN"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    games["Acute_Load_5G"] = games.groupby("player_name")["MIN"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    games["Chronic_Load_15G"] = games.groupby("player_name")["MIN"].transform(
        lambda x: x.rolling(window=15, min_periods=1).mean()
    )
    games["ACWR"] = (games["Acute_Load_5G"] / games["Chronic_Load_15G"]).replace(
        [pd.NA, pd.NaT, float("inf"), -float("inf")], 0
    )
    games["ACWR"] = games["ACWR"].fillna(0)
    return games


def add_biometric_tiers(games):
    games["Height_Tier"] = pd.cut(
        games["heightInches"],
        bins=[-float("inf"), 72, 74, 77, 80, 83, float("inf")],
        labels=["< 6ft", "6ft0 - 6ft2", "6ft3 - 6ft5", "6ft6 - 6ft8", "6ft9 - 6ft11", "7ft+"],
    )
    games["Weight_Tier"] = pd.cut(
        games["bodyWeightLbs"],
        bins=[-float("inf"), 180, 195, 210, 225, 240, 255, float("inf")],
        labels=["< 180", "180-195", "195-210", "210-225", "225-240", "240-255", "255+"],
    )
    games["BMI"] = 703 * games["bodyWeightLbs"] / (games["heightInches"] ** 2)
    games["BMI_Tier"] = pd.cut(
        games["BMI"],
        bins=[-float("inf"), 23, 25, 27, 30, float("inf")],
        labels=["<23", "23-25", "25-27", "27-30", "30+"],
    )
    position_group = pd.Series("Unknown", index=games.index)
    position_group = position_group.mask(games["center"] == 1, "Center")
    position_group = position_group.mask((games["forward"] == 1) & (games["center"] != 1), "Forward")
    position_group = position_group.mask(
        (games["guard"] == 1) & (games["forward"] != 1) & (games["center"] != 1),
        "Guard",
    )
    position_group = position_group.mask(
        (games["guard"] == 1) & (games["forward"] == 1) & (games["center"] != 1),
        "Guard-Forward",
    )
    position_group = position_group.mask(
        (games["forward"] == 1) & (games["center"] == 1),
        "Forward-Center",
    )
    games["Position_Group"] = position_group
    return games


def add_context_features(games):
    usage_cols = [col for col in ["usagePercentage", "estimatedUsagePercentage"] if col in games.columns]
    pace_cols = [col for col in ["pace", "estimatedPace"] if col in games.columns]

    usage = games[usage_cols].astype(float).mean(axis=1) if usage_cols else pd.Series(0.0, index=games.index)
    pace = games[pace_cols].astype(float).mean(axis=1) if pace_cols else pd.Series(0.0, index=games.index)

    games["Usage_Pace_Mix"] = usage * pace
    return games


def add_back_to_backs(games):
    games = games.sort_values(by=["player_name", "GAME_DATE"])
    games["Days_Since_Last_Game"] = games.groupby("player_name")["GAME_DATE"].diff().dt.days
    games["Back_to_Back"] = (games["Days_Since_Last_Game"] == 1).astype(int)
    return games


def map_injuries(games, injuries):
    injuries = injuries.copy()
    injuries["Date"] = pd.to_datetime(injuries["Date"], errors="coerce").dt.normalize()
    injuries = injuries[(injuries["Date"] >= START_DATE) & (injuries["Date"] <= END_DATE)]
    injuries["Injury_Class"] = injuries["Reason"].apply(categorize_injury)

    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="coerce").dt.normalize()
    games = games[(games["GAME_DATE"] >= START_DATE) & (games["GAME_DATE"] <= END_DATE)]

    games["player_name"] = games["player_name"].astype(str).str.strip()
    injuries["player_name"] = injuries["player_name"].astype(str).str.strip()

    def normalize_name(series):
        return series.str.lower().str.replace(r"\s+", "", regex=True)

    games["name_key"] = normalize_name(games["player_name"])
    injuries["name_key"] = normalize_name(injuries["player_name"])

    valid_names = set(games["name_key"].dropna().unique())
    injuries = injuries[injuries["name_key"].isin(valid_names)]

    games["GAME_DATE_1D"] = games["GAME_DATE"] + pd.Timedelta(days=1)
    games["GAME_DATE_2D"] = games["GAME_DATE"] + pd.Timedelta(days=2)

    injuries_1d = injuries.rename(columns={"Date": "GAME_DATE_1D"})
    injuries_2d = injuries.rename(columns={"Date": "GAME_DATE_2D"})

    merge_1d = games.merge(
        injuries_1d[["name_key", "GAME_DATE_1D", "Injury_Class"]],
        on=["name_key", "GAME_DATE_1D"],
        how="left",
    )
    merge_2d = games.merge(
        injuries_2d[["name_key", "GAME_DATE_2D", "Injury_Class"]],
        on=["name_key", "GAME_DATE_2D"],
        how="left",
    )

    games["Injury_Class"] = merge_1d["Injury_Class"].fillna(0).astype(int)
    games["Injury_Class_2D"] = merge_2d["Injury_Class"].fillna(0).astype(int)
    games["Injury_Class"] = games[["Injury_Class", "Injury_Class_2D"]].max(axis=1)

    games = games.drop(columns=["GAME_DATE_1D", "GAME_DATE_2D", "Injury_Class_2D", "name_key"])
    return games


def main():
    CLEANED_TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading merged game logs...")
    games = pd.read_csv(RAW_GAME_LOGS_PATH)
    print("Loading cleaned injuries...")
    injuries = pd.read_csv(CLEANED_INJURIES_PATH)

    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="coerce")
    games = games[(games["GAME_DATE"] >= START_DATE) & (games["GAME_DATE"] <= END_DATE)]

    print("Calculating workload features...")
    games = add_workload_features(games)

    print("Adding biometric tiers...")
    games = add_biometric_tiers(games)

    print("Adding usage/pace mix...")
    games = add_context_features(games)

    print("Adding back-to-back flags...")
    games = add_back_to_backs(games)

    print("Mapping injuries to games...")
    games = map_injuries(games, injuries)

    final_cols = [
        "player_id",
        "player_name",
        "GAME_DATE",
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
        "Injury_Class",
    ]

    final_df = games[final_cols].copy()
    final_df.to_csv(CLEANED_TRAINING_DATA_PATH, index=False)
    print(f"Saved cleaned training data to: {CLEANED_TRAINING_DATA_PATH}")
    print(f"Final dataset shape: {final_df.shape}")


if __name__ == "__main__":
    main()
