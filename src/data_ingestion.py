import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from nbainjuries import injury
import time
from config import (
    SAMPLE_PLAYER_COUNT,
    SEASONS,
    API_DELAY_SECONDS,
    INJURY_REPORT_HOUR,
    INJURY_REPORT_MINUTE,
    INJURY_API_DELAY_SECONDS,
    SEASON_START_DATE,
    RAW_GAME_LOGS_PATH,
    RAW_INJURIES_PATH,
)


def get_sample_players(n=SAMPLE_PLAYER_COUNT):
    all_players = players.get_players()
    sample = [p for p in all_players if p["is_active"]][:n]
    return sample


def fetch_game_logs(player_id, season="2025-26"):
    try:
        logs = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = logs.get_data_frames()[0]
        time.sleep(API_DELAY_SECONDS)
        return df
    except Exception as e:
        print(f"Error fetching game logs for player {player_id}: {e}")
        return pd.DataFrame()


def fetch_injury_history(start_date, end_date):
    all_injuries = []
    current = start_date
    while current <= end_date:
        try:
            ts = datetime(
                current.year,
                current.month,
                current.day,
                INJURY_REPORT_HOUR,
                INJURY_REPORT_MINUTE,
            )
            report = injury.get_reportdata(ts, return_df=True)
            if report is not None and not report.empty:
                all_injuries.append(report)
            time.sleep(INJURY_API_DELAY_SECONDS)
        except Exception:
            pass

        current += timedelta(days=1)

    if all_injuries:
        return pd.concat(all_injuries, ignore_index=True)
    return pd.DataFrame()


def main():
    print("Fetching sample players...")
    sample_players = get_sample_players(n=SAMPLE_PLAYER_COUNT)
    print(f"Got {len(sample_players)} players")

    all_game_logs = []
    for i, player in enumerate(sample_players):
        print(
            f"Fetching game logs for {player['full_name']} ({i + 1}/{len(sample_players)})"
        )
        for season in SEASONS:
            df = fetch_game_logs(player["id"], season)
            if not df.empty:
                df["player_id"] = player["id"]
                df["player_name"] = player["full_name"]
                all_game_logs.append(df)

    game_logs_df = (
        pd.concat(all_game_logs, ignore_index=True) if all_game_logs else pd.DataFrame()
    )
    print(f"Game logs shape: {game_logs_df.shape}")

    print("Fetching injury history...")
    end_date = datetime.now()
    start_date = SEASON_START_DATE
    injury_df = fetch_injury_history(start_date, end_date)
    print(f"Injury history shape: {injury_df.shape}")

    if not game_logs_df.empty:
        game_logs_df.to_csv(RAW_GAME_LOGS_PATH, index=False)
        print(f"Saved game logs shape: {game_logs_df.shape}")

    if not injury_df.empty:
        injury_df.to_csv(RAW_INJURIES_PATH, index=False)
        print(f"Saved injury history shape: {injury_df.shape}")


if __name__ == "__main__":
    main()
