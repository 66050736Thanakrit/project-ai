import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

FEATURES = ["dow", "month", "day", "is_weekend", "Promotion", "AvgPrice",
            "lag_1", "lag_7", "lag_14", "roll7", "roll28"]

FEATURE_LABELS = {
    "dow": "Day of week", "month": "Month", "day": "Day of month",
    "is_weekend": "Weekend", "Promotion": "Promotion", "AvgPrice": "Avg price",
    "lag_1": "Sales 1 day ago", "lag_7": "Sales 7 days ago", "lag_14": "Sales 14 days ago",
    "roll7": "7-day avg", "roll28": "28-day avg",
}


def _calendar(d):
    return {"dow": d.dayofweek, "month": d.month, "day": d.day, "is_weekend": int(d.dayofweek >= 5)}


def make_features(daily):
    d = daily.copy()
    d["dow"] = d["Date"].dt.dayofweek
    d["month"] = d["Date"].dt.month
    d["day"] = d["Date"].dt.day
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    for lag in (1, 7, 14):
        d[f"lag_{lag}"] = d["Sales"].shift(lag)
    d["roll7"] = d["Sales"].shift(1).rolling(7).mean()
    d["roll28"] = d["Sales"].shift(1).rolling(28).mean()
    return d.dropna().reset_index(drop=True)


def _models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1),
    }


def _score(name, y, p):
    return {"Model": name, "MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))), "R2": r2_score(y, p)}


def train_regression(daily):
    """แบ่ง train/test ตามเวลา (80/20) และเทียบกับ baseline"""
    feat = make_features(daily)
    if len(feat) < 50:
        return None
    cut = int(len(feat) * 0.8)
    tr, te = feat.iloc[:cut], feat.iloc[cut:]

    rows = [_score("Baseline (lag 7)", te["Sales"], te["lag_7"])]
    test = te[["Date", "Sales"]].rename(columns={"Sales": "Actual"}).copy()
    for name, m in _models().items():
        m.fit(tr[FEATURES], tr["Sales"])
        pred = m.predict(te[FEATURES])
        test[name] = pred
        rows.append(_score(name, te["Sales"], pred))
    metrics = pd.DataFrame(rows)
    best = metrics[metrics["Model"] != "Baseline (lag 7)"].sort_values("MAE").iloc[0]["Model"]

    # refit ด้วยข้อมูลทั้งหมดเพื่อใช้พยากรณ์อนาคต
    final = _models()[best].fit(feat[FEATURES], feat["Sales"])
    rf = _models()["Random Forest"].fit(feat[FEATURES], feat["Sales"])
    importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    corr = feat[["Sales", "Promotion", "AvgPrice", "is_weekend", "dow", "month"]].corr()["Sales"].drop("Sales")
    return {"metrics": metrics, "test": test, "best": best, "model": final,
            "importance": importance, "corr": corr, "test_size": len(te)}


def regression_forecast(model, daily, horizon, promo=0):
    """พยากรณ์ล่วงหน้าแบบ recursive โดยใช้ผลพยากรณ์เป็น lag ของวันถัดไป"""
    sales = list(daily["Sales"])
    price = float(daily["AvgPrice"].tail(28).mean())
    last = daily["Date"].max()
    out = []
    for i in range(1, horizon + 1):
        d = last + pd.Timedelta(days=i)
        row = {**_calendar(d), "Promotion": promo, "AvgPrice": price,
               "lag_1": sales[-1], "lag_7": sales[-7], "lag_14": sales[-14],
               "roll7": np.mean(sales[-7:]), "roll28": np.mean(sales[-28:])}
        pred = max(float(model.predict(pd.DataFrame([row])[FEATURES])[0]), 0.0)
        sales.append(pred)
        out.append((d, pred))
    return pd.DataFrame(out, columns=["Date", "Forecast"])


def _hw(y):
    return ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add",
                                seasonal_periods=7, initialization_method="estimated").fit()


def timeseries_forecast(daily, horizon):
    """Holt-Winters (trend + weekly seasonality) พร้อมช่วงความเชื่อมั่นโดยประมาณ"""
    y = daily.set_index("Date")["Sales"].asfreq("D")
    fit = _hw(y)
    fc = fit.forecast(horizon).clip(lower=0)
    sd = float(np.std(fit.resid, ddof=1))
    out = pd.DataFrame({"Date": fc.index, "Forecast": fc.values})
    out["Lower"] = (out["Forecast"] - 1.96 * sd).clip(lower=0)
    out["Upper"] = out["Forecast"] + 1.96 * sd

    # backtest 28 วันล่าสุดเทียบ baseline
    back = None
    if len(y) >= 70:
        train, test = y.iloc[:-28], y.iloc[-28:]
        p = _hw(train).forecast(28).clip(lower=0)
        naive = y.shift(7).iloc[-28:]
        back = {"MAE": mean_absolute_error(test, p), "Naive_MAE": mean_absolute_error(test, naive)}
    return out, back
