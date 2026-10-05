import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.goldprice.dev/v1"
API_KEY = os.getenv("GOLDPRICE_API_KEY", "").strip()
SYMBOL = "XAU-USD-SPOT"

def _headers():
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

def _get(path, params=None, timeout=20):
    r = requests.get(
        f"{BASE_URL}{path}",
        params=params or {},
        headers=_headers(),
        timeout=timeout,
    )
    if r.status_code == 429:
        raise RuntimeError("goldprice.dev rate limit reached. Please wait and try again.")
    r.raise_for_status()
    return r.json()

def get_gold_spot():
    data = _get("/prices", {"symbol": SYMBOL})
    rows = data.get("symbols", [])
    if not rows:
        raise RuntimeError(f"Unexpected goldprice.dev response: {data}")
    row = rows[0]
    return {
        "symbol": "XAUUSD",
        "timestamp": row.get("computed_at", ""),
        "price": float(row["price"]),
        "bid": float(row["bid"]) if row.get("bid") is not None else None,
        "ask": float(row["ask"]) if row.get("ask") is not None else None,
        "is_stale": bool(row.get("is_stale", False)),
    }

def get_gold_history(days=30):
    # goldprice.dev Free tier: most recent 30 days of daily XAU/USD bars.
    days = min(max(int(days), 2), 30)
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days - 1)

    data = _get(
        "/bars",
        {
            "symbol": SYMBOL,
            "interval": "1d",
            "from": start.isoformat(),
            "to": today.isoformat(),
            "limit": 100,
        },
        timeout=30,
    )
    bars = data.get("bars", [])
    if not bars:
        raise RuntimeError(f"No historical bars returned: {data}")

    df = pd.DataFrame(bars)
    df["Date"] = pd.to_datetime(df["bar_start"], errors="coerce", utc=True).dt.tz_localize(None)
    df["Gold_Price"] = pd.to_numeric(df["close"], errors="coerce")
    return (
        df[["Date", "Gold_Price"]]
        .dropna()
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )
