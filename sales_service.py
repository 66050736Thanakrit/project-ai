import numpy as np
import pandas as pd

REQUIRED = ["Date", "Product", "Quantity", "Price"]


def generate_sample(seed=42, days=730):
    """ข้อมูลยอดขายจำลอง 2 ปี (มี trend, ฤดูกาล, วันหยุด, โปรโมชัน)"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
    # ชื่อสินค้า: (หมวด, ราคา, ยอดขายพื้นฐานต่อวัน)
    products = {
        "Latte": ("Drinks", 65, 40), "Americano": ("Drinks", 55, 50),
        "Green Tea": ("Drinks", 50, 22), "Croissant": ("Bakery", 70, 25),
        "Cheesecake": ("Bakery", 95, 14), "Sandwich": ("Food", 89, 18),
        "Pasta": ("Food", 129, 10), "Coffee Beans": ("Retail", 350, 3),
    }
    rows = []
    for d in dates:
        weekend = 1.35 if d.dayofweek >= 5 else 1.0
        season = 1 + 0.15 * np.sin(2 * np.pi * d.dayofyear / 365)
        trend = 1 + 0.0006 * (d - dates[0]).days
        promo = int(rng.random() < 0.15)
        for name, (cat, price, base) in products.items():
            p = price * (0.85 if promo else 1.0)
            lam = max(base * weekend * season * trend * (1.3 if promo else 1.0), 0.5)
            qty = int(rng.poisson(lam))
            rows.append([d, name, cat, qty, round(p, 2), promo])
    return pd.DataFrame(rows, columns=["Date", "Product", "Category", "Quantity", "Price", "Promotion"])


def clean_sales(df):
    """ตรวจและทำความสะอาดไฟล์ที่อัปโหลด"""
    df = df.copy()
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for name in ["Date", "Product", "Category", "Quantity", "Price", "Sales", "Promotion"]:
        if name.lower() in lower:
            rename[lower[name.lower()]] = name
    df = df.rename(columns=rename)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"ไม่พบคอลัมน์ที่จำเป็น: {', '.join(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    if "Sales" in df.columns:
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    else:
        df["Sales"] = df["Quantity"] * df["Price"]
    if "Category" not in df.columns:
        df["Category"] = "All"
    if "Promotion" not in df.columns:
        df["Promotion"] = 0
    df["Promotion"] = pd.to_numeric(df["Promotion"], errors="coerce").fillna(0).clip(0, 1).astype(int)

    df = df.dropna(subset=["Date", "Product", "Quantity", "Price", "Sales"])
    df = df[(df["Quantity"] >= 0) & (df["Sales"] >= 0)]
    if df["Date"].nunique() < 60:
        raise ValueError("ข้อมูลควรมีอย่างน้อย 60 วัน (แนะนำ 1-2 ปี) เพื่อให้โมเดลทำงานได้")
    return df.reset_index(drop=True)


def daily_sales(df):
    """รวมเป็นยอดขายรายวัน (เติมวันที่ขาดให้ครบ)"""
    g = df.groupby("Date").agg(
        Sales=("Sales", "sum"), Quantity=("Quantity", "sum"), Promotion=("Promotion", "max")
    )
    g["AvgPrice"] = g["Sales"] / g["Quantity"].replace(0, np.nan)
    full = pd.date_range(g.index.min(), g.index.max(), freq="D")
    g = g.reindex(full)
    g.index.name = "Date"
    g[["Sales", "Quantity", "Promotion"]] = g[["Sales", "Quantity", "Promotion"]].fillna(0)
    g["AvgPrice"] = g["AvgPrice"].ffill().bfill()
    return g.reset_index()


def overall_kpis(daily):
    last = daily.tail(30)["Sales"].sum()
    prev = daily.iloc[-60:-30]["Sales"].sum()
    return {
        "total_sales": float(daily["Sales"].sum()),
        "total_qty": float(daily["Quantity"].sum()),
        "avg_daily": float(daily["Sales"].mean()),
        "last30": float(last),
        "growth_30d": float((last / prev - 1) * 100) if prev > 0 else np.nan,
        "start": daily["Date"].min().date(),
        "end": daily["Date"].max().date(),
        "days": len(daily),
    }


def product_summary(df):
    """Pareto/ABC + อัตราการเติบโต 30 วันล่าสุดเทียบ 30 วันก่อนหน้า"""
    end = df["Date"].max()
    recent = df[df["Date"] > end - pd.Timedelta(days=30)]
    prior = df[(df["Date"] <= end - pd.Timedelta(days=30)) & (df["Date"] > end - pd.Timedelta(days=60))]

    t = df.groupby(["Product", "Category"], as_index=False).agg(Sales=("Sales", "sum"), Quantity=("Quantity", "sum"))
    t = t.sort_values("Sales", ascending=False).reset_index(drop=True)
    t["Share"] = t["Sales"] / t["Sales"].sum() * 100
    t["CumShare"] = t["Share"].cumsum()
    prev_cum = t["CumShare"] - t["Share"]
    t["ABC"] = np.where(prev_cum < 80, "A", np.where(prev_cum < 95, "B", "C"))

    r = recent.groupby("Product")["Sales"].sum()
    p = prior.groupby("Product")["Sales"].sum()
    t["Growth30d"] = t["Product"].map(lambda x: (r.get(x, 0) / p[x] - 1) * 100 if p.get(x, 0) > 0 else np.nan)
    return t


def category_summary(df):
    return df.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
