# GitHub Churn Predictor

A FastAPI service that predicts whether a GitHub user has "churned" — i.e. disengaged from the platform — based on their public profile data.

## How It Works

Churn is defined as **90+ days of profile inactivity**. A Random Forest classifier is trained on engineered features derived from the GitHub API, then served via a REST API.

The five features the model uses:

| Feature | Description |
|---|---|
| `days_inactive` | Days since the profile was last updated |
| `log_followers` | log(followers + 1) — reduces skew from mega-accounts |
| `log_repos` | log(public_repos + 1) — same reason |
| `total_connections` | followers + following combined |
| `follower_ratio` | followers / (following + 1) — influence signal |

> **Note:** `days_inactive` is by far the strongest signal since it directly defines the churn label. Users under ~85 days will almost always score Low risk; users over ~95 days will almost always score High. The other features add nuance for borderline cases.

---

## Quickstart with Docker

```bash
# 1. Clone / unzip the project
cd Churn-Predictor

# 2. Build and start the API
docker-compose up --build

# 3. API is live at http://localhost:8000
```

---

## API Endpoints

### `GET /health`
```bash
curl http://localhost:8000/health
```
```json
{ "status": "ok" }
```

---

### `GET /predict/{username}`
Fetches a real GitHub user's data and returns a churn prediction.

```bash
curl http://localhost:8000/predict/torvalds
```
```json
{
  "username": "torvalds",
  "profile_url": "https://github.com/torvalds",
  "public_repos": 10,
  "followers": 236000,
  "following": 0,
  "last_updated": "2024-11-03T10:12:00Z",
  "features": {
    "days_inactive": 12,
    "log_followers": 12.37,
    "log_repos": 2.40,
    "total_connections": 236000,
    "follower_ratio": 236000.0
  },
  "prediction": {
    "churned": false,
    "churn_probability": 0.01,
    "risk_level": "Low"
  }
}
```

**More examples to try:**
```bash
# Very active developer — expect Low risk
curl http://localhost:8000/predict/sindresorhus

# Long-inactive account — expect High risk
curl http://localhost:8000/predict/why-the-lucky-stiff

# Typical contributor — expect Medium/Low
curl http://localhost:8000/predict/gvanrossum
```

---

### `POST /predict`
Send features manually if you already have them computed.

**Low risk** (active user, 30 days inactive):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "days_inactive": 30,
    "log_followers": 3.5,
    "log_repos": 3.0,
    "total_connections": 200,
    "follower_ratio": 2.1
  }'
```
```json
{ "churned": false, "churn_probability": 0.03, "risk_level": "Low" }
```

**Medium risk** (borderline — 80 days inactive):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "days_inactive": 80,
    "log_followers": 2.8,
    "log_repos": 2.2,
    "total_connections": 80,
    "follower_ratio": 1.2
  }'
```
```json
{ "churned": false, "churn_probability": 0.17, "risk_level": "Low" }
```

**High risk** (200 days inactive, small profile):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "days_inactive": 200,
    "log_followers": 0.7,
    "log_repos": 0.7,
    "total_connections": 4,
    "follower_ratio": 0.5
  }'
```
```json
{ "churned": true, "churn_probability": 1.0, "risk_level": "High" }
```

**Nuanced case** (200 days inactive but high-profile user):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "days_inactive": 200,
    "log_followers": 6.5,
    "log_repos": 4.5,
    "total_connections": 2000,
    "follower_ratio": 8.0
  }'
```
```json
{ "churned": true, "churn_probability": 0.56, "risk_level": "Medium" }
```

---

### `GET /features`
Returns the feature list the model expects.
```bash
curl http://localhost:8000/features
```

---

## Interactive Docs

FastAPI auto-generates a Swagger UI — open this in your browser while the container is running:

```
http://localhost:8000/docs
```

---

## Retraining the Model

If you want to scrape fresh GitHub data and retrain from scratch:

```bash
# Run all three steps in order
python train.py
```

Or step by step:
```bash
python app/scraper.py          # fetch users → data/raw/github_users_raw.csv
python app/features.py         # engineer features → data/raw/github_features.csv
python notebooks/analysis.py   # train & save model → app/model.pkl
```

> Add a GitHub token to `.env` (see `.env.example`) to raise the API rate limit from 60 to 5000 requests/hour.

---

## Project Structure

```
Churn-Predictor/
├── app/
│   ├── main.py           # FastAPI routes
│   ├── model.py          # loads pkl files, exposes predict()
│   ├── features.py       # feature engineering logic
│   ├── scraper.py        # GitHub API scraper
│   ├── model.pkl         # trained Random Forest
│   ├── scaler.pkl        # fitted StandardScaler
│   └── features_list.pkl # ordered feature names
├── data/raw/
│   ├── github_users_raw.csv   # raw scraped data (250 users)
│   └── github_features.csv    # engineered features
├── notebooks/
│   └── analysis.py       # feature selection + model training
├── train.py              # convenience script to retrain
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
