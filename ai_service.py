import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

DOW = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]


def _n(v, s="", d=2):
    return "N/A" if v is None or pd.isna(v) else f"{v:,.{d}f}{s}"


def build_context(kpis, products, daily, reg, ts_fc, backtest):
    """สรุปตัวเลขสำคัญเป็นข้อความ (คำนวณด้วยโค้ด ไม่ให้ LLM คำนวณเอง)"""
    L = [
        f"ช่วงข้อมูล: {kpis['start']} ถึง {kpis['end']} ({kpis['days']} วัน)",
        f"ยอดขายรวม: {_n(kpis['total_sales'])} | เฉลี่ยต่อวัน: {_n(kpis['avg_daily'])}",
        f"ยอดขาย 30 วันล่าสุด: {_n(kpis['last30'])} (เทียบ 30 วันก่อนหน้า: {_n(kpis['growth_30d'], '%')})",
    ]

    dow = daily.groupby(daily["Date"].dt.dayofweek)["Sales"].mean()
    L.append(f"วันที่ขายดีสุด: {DOW[int(dow.idxmax())]} ({_n(dow.max())}), "
             f"วันที่ขายต่ำสุด: {DOW[int(dow.idxmin())]} ({_n(dow.min())})")

    promo = daily.groupby("Promotion")["Sales"].mean()
    if 0 in promo.index and 1 in promo.index and promo[0] > 0:
        L.append(f"ยอดขายเฉลี่ยวันมีโปรโมชัน {_n(promo[1])} vs วันปกติ {_n(promo[0])} "
                 f"({_n((promo[1] / promo[0] - 1) * 100, '%')})")

    L.append("สินค้า (ยอดขาย | สัดส่วน | ABC | เติบโต 30 วัน):")
    for _, r in products.head(6).iterrows():
        L.append(f"- {r['Product']} ({r['Category']}): {_n(r['Sales'], d=0)} | "
                 f"{_n(r['Share'], '%', 1)} | {r['ABC']} | {_n(r['Growth30d'], '%', 1)}")
    g = products.dropna(subset=["Growth30d"]).sort_values("Growth30d")
    if len(g):
        L.append(f"โตเร็วสุด: {g.iloc[-1]['Product']} ({_n(g.iloc[-1]['Growth30d'], '%', 1)}), "
                 f"ลดลงมากสุด: {g.iloc[0]['Product']} ({_n(g.iloc[0]['Growth30d'], '%', 1)})")

    if reg:
        m = reg["metrics"].set_index("Model")
        L.append("โมเดล Regression (test แบบแบ่งตามเวลา):")
        for name, r in m.iterrows():
            L.append(f"- {name}: MAE {_n(r['MAE'], d=0)}, RMSE {_n(r['RMSE'], d=0)}, R2 {_n(r['R2'], d=3)}")
        top = reg["importance"].head(3)
        L.append("ปัจจัยสำคัญสุด: " + ", ".join(f"{k} ({v:.0%})" for k, v in top.items()))

    if ts_fc is not None:
        L.append(f"พยากรณ์ Time Series {len(ts_fc)} วันข้างหน้า: รวม {_n(ts_fc['Forecast'].sum(), d=0)}, "
                 f"เฉลี่ย/วัน {_n(ts_fc['Forecast'].mean(), d=0)}")
    if backtest:
        L.append(f"Backtest 28 วัน: MAE Holt-Winters {_n(backtest['MAE'], d=0)} vs baseline {_n(backtest['Naive_MAE'], d=0)}")
    return "\n".join(L)


def ask_sales_ai(question, context):
    if not KEY:
        raise ValueError("Missing GEMINI_API_KEY in .env")
    prompt = f"""ข้อมูลสรุปยอดขายของร้านค้า:
{context}

คำถาม: {question}

ตอบเป็นภาษาไทย โดยอ้างอิงตัวเลขที่ให้เท่านั้น
จัดโครงสร้างเป็น: สรุปผล, สาเหตุที่เป็นไปได้ (ระบุว่าเป็นข้อสันนิษฐานจากข้อมูล), ข้อเสนอแนะเชิงธุรกิจที่ทำได้จริง
ห้ามสร้างตัวเลข ข่าว หรือปัจจัยภายนอกที่ไม่มีในข้อมูล และไม่รับประกันยอดขายในอนาคต"""

    client = genai.Client(api_key=KEY)
    last = None
    for delay in [0, 2, 4]:
        if delay:
            time.sleep(delay)
        try:
            res = client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a concise retail sales analyst who gives practical business advice.",
                    temperature=0.3))
            return res.text
        except Exception as e:
            last = e
            if "503" not in str(e) and "UNAVAILABLE" not in str(e):
                raise
    raise last
