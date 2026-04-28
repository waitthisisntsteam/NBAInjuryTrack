import pandas as pd

from config import (
    KAGGLE_GAME_LOGS_PATH,
    KAGGLE_PLAYERS_PATH,
    KAGGLE_INJURIES_PATH,
    RAW_GAME_LOGS_PATH,
    CLEANED_INJURIES_PATH,
    ACTIVE_PLAYERS_PATH,
)


START_DATE = pd.Timestamp("2016-01-01")
END_DATE = pd.Timestamp("2025-12-31")
ACTIVE_START_DATE = pd.Timestamp("2024-01-01")


def build_player_name(df):
    return df["firstName"].astype(str).str.strip() + " " + df["lastName"].astype(str).str.strip()


def prep_game_logs():
    games = pd.read_csv(KAGGLE_GAME_LOGS_PATH, low_memory=False)
    games["player_name"] = build_player_name(games)
    games = games.rename(
        columns={
            "personId": "player_id",
            "gameDateTimeEst": "GAME_DATE",
            "numMinutes": "MIN",
        }
    )

    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="coerce")
    games["MIN"] = pd.to_numeric(games["MIN"], errors="coerce")

    games = games[(games["GAME_DATE"] >= START_DATE) & (games["GAME_DATE"] <= END_DATE)]
    games = games[games["MIN"].notna() & (games["MIN"] > 0)]

    active_players = (
        games[games["GAME_DATE"] >= ACTIVE_START_DATE]
        .loc[:, "player_name"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    active_players.to_csv(ACTIVE_PLAYERS_PATH, index=False, header=["player_name"])

    return games


def prep_biometrics():
    players = pd.read_csv(KAGGLE_PLAYERS_PATH)
    players = players.rename(columns={"personId": "player_id"})
    return players[["player_id", "heightInches", "bodyWeightLbs", "guard", "forward", "center"]]


def prep_injuries():
    injuries = pd.read_csv(KAGGLE_INJURIES_PATH)
    injuries = injuries[injuries["Relinquished"].notna()].copy()
    injuries = injuries.rename(
        columns={
            "Date": "Date",
            "Relinquished": "player_name",
            "Notes": "Reason",
        }
    )
    injuries["player_name"] = injuries["player_name"].astype(str).str.split(" / ").str[0].str.strip()
    injuries["Date"] = pd.to_datetime(injuries["Date"], errors="coerce")
    injuries = injuries[(injuries["Date"] >= START_DATE) & (injuries["Date"] <= END_DATE)]
    reason_lower = injuries["Reason"].astype(str).str.lower()
    noise_keywords = [
        "rest",
        "illness",
        "covid",
        "protocol",
        "flu",
        "sick",
        "personal",
        "health and safety",
        "suspension",
        "injury management",
        "load management",
        "management",
        "conditioning",
    ]
    noise_mask = reason_lower.str.contains("|".join(noise_keywords), regex=True, na=False)
    injuries = injuries[~noise_mask]
    return injuries[["Date", "player_name", "Reason"]]


def main():
    RAW_GAME_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEANED_INJURIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_PLAYERS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Preparing Kaggle game logs...")
    games = prep_game_logs()
    print(f"Game logs rows: {len(games):,}")

    print("Preparing biometrics...")
    biometrics = prep_biometrics()
    print(f"Biometrics rows: {len(biometrics):,}")

    print("Preparing injuries...")
    injuries = prep_injuries()
    print(f"Injuries rows: {len(injuries):,}")

    merged = games.merge(biometrics, on="player_id", how="left")
    merged.to_csv(RAW_GAME_LOGS_PATH, index=False)
    injuries.to_csv(CLEANED_INJURIES_PATH, index=False)

    print(f"Saved merged game logs to: {RAW_GAME_LOGS_PATH}")
    print(f"Saved cleaned injuries to: {CLEANED_INJURIES_PATH}")
    print(f"Saved active players list to: {ACTIVE_PLAYERS_PATH}")


if __name__ == "__main__":
    main()
