import numpy as np
import pandas as pd
def calculate_statistics(df):
    if df is None or len(df)<2: raise ValueError("At least 2 observations are required.")
    data=df.copy().sort_values("Date").reset_index(drop=True)
    data["Daily_Return"]=data["Gold_Price"].pct_change()*100
    data["MA7"]=data["Gold_Price"].rolling(7).mean()
    data["MA30"]=data["Gold_Price"].rolling(30).mean()
    p=data["Gold_Price"]; r=data["Daily_Return"].dropna()
    s={"mean":float(p.mean()),"median":float(p.median()),"std":float(p.std(ddof=1)),
       "min":float(p.min()),"max":float(p.max()),"range":float(p.max()-p.min()),
       "latest_daily_return":float(data["Daily_Return"].iloc[-1]),
       "period_return":float((p.iloc[-1]/p.iloc[0]-1)*100),
       "average_daily_return":float(r.mean()),"daily_volatility":float(r.std(ddof=1)),
       "ma7":float(data["MA7"].iloc[-1]) if pd.notna(data["MA7"].iloc[-1]) else np.nan,
       "ma30":float(data["MA30"].iloc[-1]) if pd.notna(data["MA30"].iloc[-1]) else np.nan}
    return s,data
