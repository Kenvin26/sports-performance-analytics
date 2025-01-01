import json
import pandas as pd
import os

def process_match_data():
    # Load match data from JSON
    with open("data/matches.json", "r") as f:
        matches = json.load(f)

    # Extract relevant fields
    match_list = []
    for match in matches.get("matches", []):
        match_list.append({
            "match_id": match.get("id"),
            "date": match.get("utcDate"),
            "home_team": match.get("homeTeam", {}).get("name"),
            "away_team": match.get("awayTeam", {}).get("name"),
            "home_score": match.get("score", {}).get("fullTime", {}).get("homeTeam"),
            "away_score": match.get("score", {}).get("fullTime", {}).get("awayTeam"),
            "winner": match.get("score", {}).get("winner")  # HOME_TEAM, AWAY_TEAM, or DRAW
        })

    # Create DataFrame
    df = pd.DataFrame(match_list)

    # Save processed data
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/processed_matches.csv", index=False)
    print("Processed data saved to 'data/processed_matches.csv'!")

if __name__ == "__main__":
    process_match_data()
