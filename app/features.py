import pandas as pd
import numpy as np

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw GitHub user data into meaningful ML features.
    Each feature is a hypothesis about what signals churn risk.
    """
    features = pd.DataFrame()

    # --- RATIO FEATURES (≥2 required) ---
    # Follower ratio: high ratio = influential user, less likely to churn
    features["follower_ratio"] = (
        df["followers"] / (df["following"] + 1)
    )

    # Repos per year: normalizes productivity by account age
    features["repos_per_year"] = (
        df["public_repos"] / (df["account_age_days"] / 365 + 1)
    )

    # Gists per repo: users who share code snippets are more engaged
    features["gists_per_repo"] = (
        df["public_gists"] / (df["public_repos"] + 1)
    )

    # --- TIME-BASED FEATURES (≥2 required) ---
    # Days inactive: primary churn signal — recency of last activity
    features["days_inactive"] = df["days_inactive"]

    # Account age in years: older accounts are less likely to churn
    features["account_age_years"] = df["account_age_days"] / 365

    # --- AGGREGATION FEATURES (≥2 required) ---
    # Log followers: reduces skew from mega-accounts
    features["log_followers"] = np.log1p(df["followers"])

    # Total social connections: followers + following combined
    features["total_connections"] = df["followers"] + df["following"]

    # Log repos: reduces skew from very prolific developers
    features["log_repos"] = np.log1p(df["public_repos"])

    # --- BINARY / CATEGORICAL FEATURES (≥2 required) ---
    # Has no repos: users with 0 repos never really engaged
    features["has_no_repos"] = (df["public_repos"] == 0).astype(int)

    # Has bio: users who filled out their profile are more committed
    features["has_bio"] = df["has_bio"].astype(int)

    # Has blog: external blog = strong engagement signal
    features["has_blog"] = df["has_blog"].astype(int)

    # Has company: professional users churn less
    features["has_company"] = df["has_company"].astype(int)

    # Keep username for reference
    features["username"] = df["username"].values

    return features


def label_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Churn definition: a GitHub user is considered churned if their
    profile has not been updated in more than 90 days.
    Threshold justification: 90 days (one quarter) of inactivity
    is a strong signal that a developer has disengaged from the platform.
    """
    df["churned"] = (df["days_inactive"] > 90).astype(int)
    return df


if __name__ == "__main__":
    raw = pd.read_csv("data/raw/github_users_raw.csv")
    raw = label_churn(raw)

    print("=== Class Balance ===")
    print(raw["churned"].value_counts())
    print(f"Churn rate: {raw['churned'].mean():.1%}")

    features = generate_features(raw)
    features["churned"] = raw["churned"].values

    print("\n=== Feature Preview ===")
    print(features.head(10).to_string())

    print(f"\nShape: {features.shape}")
    print(f"\nMissing values:\n{features.isnull().sum()}")

    features.to_csv("data/raw/github_features.csv", index=False)
    print("\n Features saved to data/raw/github_features.csv")