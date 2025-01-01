import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the trained model
MODEL_PATH = Path("models/match_prediction_model.pkl")
model = joblib.load(MODEL_PATH)

# Function to preprocess user input
def preprocess_input(home_team_rank, away_team_rank, expected_goal_diff, home_strength, home_advantage, away_challenge, is_weekend, is_evening_game, days_from_start, day_of_week, month):
    data = {
        'home_team_rank': [home_team_rank],
        'away_team_rank': [away_team_rank],
        'expected_goal_diff': [expected_goal_diff],
        'home_strength': [home_strength],
        'home_advantage': [home_advantage],
        'away_challenge': [away_challenge],
        'is_weekend': [is_weekend],
        'is_evening_game': [is_evening_game],
        'days_from_start': [days_from_start],
        'day_of_week': [day_of_week],
        'month': [month],
    }
    return pd.DataFrame(data)

# Streamlit app UI
st.title("Football Match Outcome Predictor")
st.markdown("Predict the outcome of a football match: **Home Win**, **Away Win**, or **Draw**.")

# User inputs
home_team_rank = st.slider("Home Team Rank", 1, 5, 3)
away_team_rank = st.slider("Away Team Rank", 1, 5, 3)
expected_goal_diff = st.number_input("Expected Goal Difference", value=0.0, step=0.1)
home_strength = st.number_input("Home Strength", value=3.0, step=0.1)
home_advantage = st.number_input("Home Advantage", value=3.0, step=0.1)
away_challenge = st.number_input("Away Challenge", value=3.0, step=0.1)
is_weekend = st.selectbox("Is it a Weekend?", ["No", "Yes"])
is_evening_game = st.selectbox("Is it an Evening Game?", ["No", "Yes"])
days_from_start = st.number_input("Days from Season Start (Normalized)", value=0.5, step=0.01)
day_of_week = st.selectbox("Day of the Week", [0, 1, 2, 3, 4, 5, 6])
month = st.selectbox("Month", list(range(1, 13)))

# Predict button
if st.button("Predict Match Outcome"):
    # Preprocess input data
    X_input = preprocess_input(
        home_team_rank, away_team_rank, expected_goal_diff,
        home_strength, home_advantage, away_challenge,
        1 if is_weekend == "Yes" else 0,
        1 if is_evening_game == "Yes" else 0,
        days_from_start, day_of_week, month
    )
    
    # Make prediction
    prediction = model.predict(X_input)
    probabilities = model.predict_proba(X_input)

    # Map predictions to labels
    outcome_mapping = {0: "Home Win", 1: "Away Win", 2: "Draw"}
    predicted_outcome = outcome_mapping[prediction[0]]

    st.subheader("Prediction Results:")
    st.write(f"**Predicted Outcome:** {predicted_outcome}")
    st.write("**Prediction Probabilities:**")
    st.write(f"- Home Win: {probabilities[0][0]:.2f}")
    st.write(f"- Away Win: {probabilities[0][1]:.2f}")
    st.write(f"- Draw: {probabilities[0][2]:.2f}")
