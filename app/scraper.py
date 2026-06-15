import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.github.com"

def fetch_github_users(usernames: list) -> pd.DataFrame:
    records = []

    for username in usernames:
        try:
            r = requests.get(
                f"{BASE_URL}/users/{username}",
                headers={"Accept": "application/vnd.github+json"}
            )

            if r.status_code == 404:
                print(f"  ️  Not found: {username}")
                continue
            if r.status_code == 403:
                print(f"  ⏳ Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                continue

            data = r.json()

            records.append({
                "username":       data.get("login", username),
                "public_repos":   data.get("public_repos", 0),
                "public_gists":   data.get("public_gists", 0),
                "followers":      data.get("followers", 0),
                "following":      data.get("following", 0),
                "created_at":     data.get("created_at", None),
                "updated_at":     data.get("updated_at", None),
                "has_bio":        int(data.get("bio") is not None),
                "has_blog":       int(bool(data.get("blog", ""))),
                "has_company":    int(bool(data.get("company", ""))),
                "hireable":       int(bool(data.get("hireable", False))),
            })

            print(f"   Fetched: {username} | repos: {data.get('public_repos')} | followers: {data.get('followers')}")

        except Exception as e:
            print(f"   Error fetching {username}: {e}")

        time.sleep(0.5)  # respect rate limits

    return pd.DataFrame(records)


def label_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Churn definition: a GitHub user is considered churned if their
    profile has not been updated in more than n days.
    This simulates a developer who has stopped being active on the platform.
    """
    now = datetime.now(timezone.utc)

    df["last_active"] = pd.to_datetime(df["updated_at"], utc=True)
    df["days_inactive"] = (now - df["last_active"]).dt.days
    df["account_age_days"] = (
        now - pd.to_datetime(df["created_at"], utc=True)
    ).dt.days

    df["churned"] = (df["days_inactive"] > 90).astype(int)
    return df


if __name__ == "__main__":
    usernames = [
        # Highly active maintainers / org leads
        "torvalds", "gvanrossum", "yyx990803", "tj", "sindresorhus",
        "addyosmani", "paulirish", "kentcdodds", "getify", "dhh",
        "mdo", "fat", "jashkenas", "isaacs", "substack",
        "creationix", "feross", "mikeal", "dominictarr", "rvagg",
        "juliangarnier", "Rich-Harris", "antfu", "patak-dev", "pi0",
        "atinux", "Akryum", "evanw", "jamiebuilds", "kittygiraudel",
        "csswizardry", "chriscoyier", "mathiasbynens", "LeaVerou", "davidwalshblog",
        "rachelandrew", "heydonworks", "adactio", "wesbos", "samuelcolvin",
        "tiangolo", "encode", "miguelgrinberg", "ines", "honnibal",
        "jakevdp", "wesm", "mwaskom", "fchollet", "soumith",
        "karpathy", "jeffheaton", "ageron", "amueller", "rasbt",

        # Mixed activity
        "mojombo", "defunkt", "pjhyett", "wycats", "ezmobius",
        "topfunky", "anotherjesse", "drnic", "hassox", "fnando",
        "jnunemaker", "technoweenie", "josevalim", "rafaelfranca", "tenderlove",
        "indirect", "spastorino", "matthewd", "pixeltrix", "jeremy",
        "jamis", "nicksieger", "hornbeck", "schneems", "samsaffron",
        "drogus", "brynary", "qrush", "phinze", "pat",
        "radar", "joshpeak", "amatsuda", "jeg2", "judofyr",
        "ConradIrwin", "norman", "lukeasrodgers", "lifo", "rkh",
        "sferik", "errfree", "wycats", "yui-knk", "eregon",
        "vmg", "tarcieri", "mperham", "skorks", "tomafro",

        # Likely inactive / historic accounts
        "bmizerany", "rtomayko", "chneukirchen", "why-the-lucky-stiff",
        "jimweirich", "seattlerb", "jbarnette", "ujihisa", "matz",
        "nobu", "ko1", "akr", "duerst", "yugui",
        "Constellation", "brixen", "headius", "enebo", "jruby",
        "nahi", "ola-bini", "MenTaLguY", "evanphx", "soba",
        "zenspider", "halorgium", "igrigorik", "wmorgan", "sporkmonger",
        "dbalatero", "raggi", "lukeredpath", "alindeman", "joshk",
        "viatropos", "railsjazz", "ryanb", "thoughtbot", "joshuaclayton",
        "jferris", "stevenharman", "mojavelinux", "burkelibbey", "myronmarston",

        # Open source / data science / web
        "numpy", "scipy", "pandas-dev", "matplotlib", "scikit-learn",
        "jupyter", "ipython", "django", "pallets", "flask",
        "psf", "pypa", "python", "nodejs", "expressjs",
        "facebook", "vercel", "remix-run", "sveltejs", "vuejs",
        "angular", "reactjs", "preactjs", "solidjs", "qwikdev",
        "tailwindlabs", "shadcn", "radix-ui", "framer", "chakra-ui",
        "emotion-js", "styled-components", "vitejs", "esbuild", "swc-project",
        "biomejs", "denoland", "oven-sh", "bunjs", "gulpjs",
        "gruntjs", "webpack", "rollup", "parcel-bundler", "browserify",

        # More individuals
        "addaleax", "BridgeAR", "bnoordhuis", "bzoz", "cjihrig",
        "danbev", "evanlucas", "fhinkel", "gibfahn", "jasnell",
        "joyeecheung", "mcollina", "MylesBorins", "ronag", "RobinTail",
        "targos", "TimothyGu", "trevnorris", "vsemozhetbyt", "watilde",
        "Trott", "apapirovski", "lpinca", "mhdawson", "mscdex",
        "ChALkeR", "shigeki", "yorkie", "thefourtheye", "santigimeno",
        "indutny", "Fishrock123", "rvagg", "piscisaureus", "tjfontaine",
        "saghul", "AndreasMadsen", "brendanashworth", "calvinmetcalf", "ChrisAlderson",

        # Additional diverse profiles
        "octocat", "github", "actions", "dependabot", "renovate-bot",
        "greenkeeperio-bot", "snyk-bot", "codecov", "travis-ci", "circleci",
        "appveyor", "netlify", "cloudflare", "fly-apps", "render-oss",
        "supabase", "planetscale", "neondatabase", "prisma", "drizzle-team",
        "trpc", "tanstack", "remult", "hasura", "graphql",
        "apollographql", "urql-graphql", "relayjs", "gqlgen", "graphile",
    ]

    print(" Fetching GitHub user data...")
    df = fetch_github_users(usernames)
    df = label_churn(df)
    df.to_csv("data/raw/github_users_raw.csv", index=False)
    print(f"\n Saved {len(df)} users to data/raw/github_users_raw.csv")
    print(f"\nChurn rate: {df['churned'].mean():.1%}")
    print(df[["username", "public_repos", "followers", "days_inactive", "churned"]].head(10).to_string())