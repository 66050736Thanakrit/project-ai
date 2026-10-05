import os, time
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()
KEY=os.getenv("GEMINI_API_KEY")
MODEL=os.getenv("GEMINI_MODEL","gemini-3.6-flash")
def f(v,s=""): return "N/A" if v is None or pd.isna(v) else f"{v:,.4f}{s}"
def ask_gold_ai(question,period,spot,stats,df):
    if not KEY: raise ValueError("Missing GEMINI_API_KEY in .env")
    context=f"""XAU/USD market statistics
Period: {period}
From: {df.iloc[0]['Date'].date()} To: {df.iloc[-1]['Date'].date()}
Observations: {len(df)}
Spot: ${spot['price']:,.2f}
Latest close: ${df.iloc[-1]['Gold_Price']:,.2f}
Mean: ${f(stats['mean'])}
Median: ${f(stats['median'])}
SD: ${f(stats['std'])}
Low: ${f(stats['min'])}
High: ${f(stats['max'])}
Latest daily return: {f(stats['latest_daily_return'],'%')}
Period return: {f(stats['period_return'],'%')}
Average daily return: {f(stats['average_daily_return'],'%')}
Daily volatility: {f(stats['daily_volatility'],'%')}
MA7: ${f(stats['ma7'])}
MA30: ${f(stats['ma30'])}"""
    prompt=f"""{context}

Question: {question}

อธิบายเป็นภาษาไทยโดยอ้างอิงตัวเลขที่ให้เท่านั้น
ห้ามสร้างข่าวหรือปัจจัยภายนอกที่ไม่มีในข้อมูล
ไม่ให้คำสั่งซื้อหรือขาย และไม่รับประกันทิศทางราคาในอนาคต"""
    client=genai.Client(api_key=KEY)
    last=None
    for delay in [0,2,4]:
        if delay: time.sleep(delay)
        try:
            res=client.models.generate_content(
                model=MODEL,contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a concise XAU/USD market data analyst.",
                    temperature=0.2))
            return res.text
        except Exception as e:
            last=e
            if "503" not in str(e) and "UNAVAILABLE" not in str(e): raise
    raise last
