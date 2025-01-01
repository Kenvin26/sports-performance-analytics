import requests
import json
import os

# Your API token
API_KEY = "cc617cbd60ae4ab2b49193e206e7db2b"  # Replace with your actual token
BASE_URL = "https://api.football-data.org/v4/"

def fetch_matches(competition_id):
    headers = {
        "X-Auth-Token": API_KEY  # Add the API token as a header
    }
    url = f"{BASE_URL}competitions/{competition_id}/matches"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        matches = response.json()
        # Save the response data into a JSON file
        os.makedirs("data", exist_ok=True)
        with open("data/matches.json", "w") as f:
            json.dump(matches, f, indent=4)
        print("Match data saved successfully!")
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")

if __name__ == "__main__":
    competition_id = "PL"  # Use the appropriate competition ID for v4 (e.g., "PL" for Premier League)
    fetch_matches(competition_id)
