import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib
import os
from pathlib import Path

def create_advanced_features(df):
    """Create advanced features for match prediction."""
    # Team strength tiers with more granular categorization
    elite_tier = ['Manchester City FC', 'Arsenal FC']
    top_tier = ['Manchester United FC', 'Liverpool FC', 'Chelsea FC', 'Tottenham Hotspur FC']
    upper_mid_tier = ['Newcastle United FC', 'Brighton & Hove Albion FC', 'Aston Villa FC']
    mid_tier = ['West Ham United FC', 'Crystal Palace FC', 'Brentford FC']
    
    # Create more granular team rankings (1-5 scale)
    def get_team_rank(team):
        if team in elite_tier: return 5
        if team in top_tier: return 4
        if team in upper_mid_tier: return 3
        if team in mid_tier: return 2
        return 1

    # Team strength features
    df['home_team_rank'] = df['home_team'].apply(get_team_rank)
    df['away_team_rank'] = df['away_team'].apply(get_team_rank)
    
    # Historical performance features (based on team ranks)
    df['expected_goal_diff'] = (df['home_team_rank'] - df['away_team_rank']) * 0.5
    df['home_strength'] = df['home_team_rank'] * 1.1  # Home advantage multiplier
    
    # Temporal features
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_evening_game'] = (df['date'].dt.hour >= 17).astype(int)
    
    # Create interaction features
    df['home_advantage'] = df['home_team_rank'] * df['is_weekend'] * 1.1
    df['away_challenge'] = df['away_team_rank'] * (2 - df['is_weekend']) * 0.9
    
    # Distance from season start (normalized)
    season_start = df['date'].min()
    df['days_from_start'] = (df['date'] - season_start).dt.days
    df['days_from_start'] = df['days_from_start'] / df['days_from_start'].max()
    
    return df

def load_and_prepare_data(file_path):
    """Load and prepare data with advanced feature engineering."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    print("\nInitial data info:")
    print(f"Total matches: {len(df)}")
    print(f"Matches with predictions: {df['winner'].notna().sum()}")
    
    # Filter to matches with predictions
    df = df[df['winner'].notna()].copy()
    
    # Create advanced features
    df = create_advanced_features(df)
    
    # Select features for model
    features = [
        'home_team_rank', 'away_team_rank', 'expected_goal_diff',
        'home_strength', 'home_advantage', 'away_challenge',
        'is_weekend', 'is_evening_game', 'days_from_start',
        'day_of_week', 'month'
    ]
    
    X = df[features]
    y = df["winner"].map({"HOME_TEAM": 0, "AWAY_TEAM": 1, "DRAW": 2})
    
    print("\nFeatures used:")
    for feature in features:
        print(f"- {feature}")
    
    print(f"\nClass distribution:")
    print(df['winner'].value_counts())
    
    return X, y, df

def create_ensemble_model():
    """Create an ensemble of multiple models."""
    # Define base models
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb)
        ],
        voting='soft'
    )
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('model', ensemble)
    ])

def train_model(X_train, y_train):
    """Train the model with cross-validation."""
    model = create_ensemble_model()
    
    # Define CV strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Fit model
    print("\nTraining ensemble model...")
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model with detailed metrics."""
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Home Win", "Away Win", "Draw"]))
    
    # Print prediction probabilities for test set
    y_proba = model.predict_proba(X_test)
    print("\nAverage prediction probabilities:")
    print("Home Win:", y_proba[:, 0].mean())
    print("Away Win:", y_proba[:, 1].mean())
    print("Draw:", y_proba[:, 2].mean())
    
    return y_pred

def main():
    data_path = Path("data/processed_matches.csv")
    model_path = Path("models/match_prediction_model.pkl")
    
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("Loading and preparing data...")
        X, y, df = load_and_prepare_data(data_path)
        
        print("\nSplitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        
        joblib.dump(model, model_path)
        print(f"\nModel saved to '{model_path}'")
        print("\nModel training completed successfully!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()