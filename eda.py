import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda():
    # Load processed data
    df = pd.read_csv("data/processed_matches.csv")

    # Show basic statistics
    print("Basic Statistics:")
    print(df.describe())

    # Plot win/loss/draw distribution
    result_counts = df["winner"].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=result_counts.index, y=result_counts.values, palette="viridis")
    plt.title("Match Results Distribution")
    plt.ylabel("Number of Matches")
    plt.xlabel("Result")
    plt.show()

    # Plot goal distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["home_score"], kde=True, color="blue", label="Home Goals", alpha=0.7, bins=15)
    sns.histplot(df["away_score"], kde=True, color="red", label="Away Goals", alpha=0.7, bins=15)
    plt.legend()
    plt.title("Goal Distribution")
    plt.xlabel("Goals Scored")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    perform_eda()
