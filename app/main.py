from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
import model as ml

PARAGUAY_TZ = ZoneInfo("America/Asuncion")

app = FastAPI(title="GitHub Churn Predictor API")


class UserFeatures(BaseModel):
    days_inactive: float
    account_age_years: float
    total_connections: float
    log_repos: float
    log_followers: float


@app.post("/predict")
def predict_churn(user: UserFeatures):
    result = ml.predict(user.dict())
    return result


@app.get("/predict/{username}")
def predict_from_github(username: str):
    """
    Fetch a real GitHub user's data and predict their churn risk.
    """
    r = requests.get(f"https://api.github.com/users/{username}")

    if r.status_code == 404:
        raise HTTPException(status_code=404, detail=f"GitHub user '{username}' not found")
    if r.status_code == 403:
        raise HTTPException(status_code=429, detail="GitHub API rate limit reached, try again later")

    data = r.json()

    now = datetime.now(PARAGUAY_TZ)
    updated_at = datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")).astimezone(PARAGUAY_TZ)
    created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")).astimezone(PARAGUAY_TZ)

    days_inactive = (now - updated_at).days
    account_age_years = (now - created_at).days / 365
    public_repos = data.get("public_repos", 0)
    followers = data.get("followers", 0)
    following = data.get("following", 0)

    features = {
        "days_inactive": days_inactive,
        "account_age_years": account_age_years,
        "total_connections": followers + following,
        "log_repos": np.log1p(public_repos),
        "log_followers": np.log1p(followers),
    }

    result = ml.predict(features)

    return {
        "username": data["login"],
        "profile_url": data["html_url"],
        "public_repos": public_repos,
        "followers": followers,
        "following": following,
        "last_updated": data["updated_at"],
        "features": features,
        "prediction": result
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    return {"expected_features": ml.feature_names}