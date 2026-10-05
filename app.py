import time
import base64
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from gold_service import get_gold_spot, get_gold_history
from stats_service import calculate_statistics
from ai_service import ask_gold_ai
from firebase_auth import register_user, login_user

def get_bg_image():
    path = Path(__file__).parent / "assets" / "gold-background.png"
    return base64.b64encode(path.read_bytes()).decode()

BG_IMAGE = get_bg_image()

st.set_page_config(
    page_title="Gold Analytics",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# XM-inspired trading UI (original design, no copied branding)
# =========================================================
st.markdown(f"""
<style>
:root{{
  --bg:#09090b;
  --surface:#111113;
  --surface2:#17171a;
  --surface3:#1d1d21;
  --border:#29292e;
  --red:#e32636;
  --red2:#b91c2b;
  --green:#16c784;
  --danger:#ea3943;
  --text:#f7f7f8;
  --muted:#96969f;
}}
html, body, [class*="css"] {{ font-family: Inter, Arial, sans-serif; }}
.stApp {{
  color:var(--text);
  background:#09090b;
  position:relative;
}}
.gold-fixed-bg {{
  position:fixed;
  inset:0;
  width:100vw;
  height:100vh;
  object-fit:cover;
  object-position:center top;
  z-index:-2;
  opacity:.58;
  pointer-events:none;
}}
.gold-bg-overlay {{
  position:fixed;
  inset:0;
  z-index:-1;
  pointer-events:none;
  background:
    linear-gradient(180deg, rgba(9,9,11,.22) 0%, rgba(9,9,11,.55) 46%, rgba(9,9,11,.88) 82%, rgba(9,9,11,.96) 100%),
    linear-gradient(90deg, rgba(9,9,11,.68) 0%, rgba(9,9,11,.40) 52%, rgba(9,9,11,.24) 100%);
}}
[data-testid="stAppViewContainer"] {{
  background:transparent !important;
  position:relative;
  isolation:isolate;
}}
[data-testid="stAppViewContainer"] > .main {{
  position:relative;
  z-index:1;
  background:transparent !important;
}}
[data-testid="stMain"] {{
  background:transparent !important;
}}
.block-container {{
  position:relative;
  z-index:2;
}}

[data-testid="stHeader"] {{ background:transparent; height:0px; }}
[data-testid="stToolbar"] {{ display:none; }}
[data-testid="stDecoration"] {{ display:none; }}
#MainMenu, footer {{ visibility:hidden; }}
.block-container{{
  max-width:1480px;
  padding-top:1.1rem;
  padding-bottom:3rem;
}}
h1,h2,h3{{letter-spacing:-.025em}}
hr{{border-color:var(--border)!important}}

/* tabs become top navigation */
button[data-baseweb="tab"]{{
  font-weight:650!important;
  color:#aaaab2!important;
}}
button[data-baseweb="tab"][aria-selected="true"]{{
  color:#fff!important;
}}
div[data-baseweb="tab-highlight"]{{
  background-color:var(--red)!important;
}}

/* metrics */
div[data-testid="stMetric"]{{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:12px;
  padding:15px 17px;
}}
div[data-testid="stMetricLabel"]{{color:var(--muted)}}
div[data-testid="stMetricValue"]{{font-size:1.42rem}}
div[data-testid="stMetricDelta"]{{font-weight:700}}

/* buttons */
.stButton > button{{
  border-radius:8px;
  min-height:40px;
  border:1px solid var(--border);
  background:var(--surface2);
  color:#fff;
  font-weight:650;
}}
.stButton > button:hover{{
  border-color:var(--red);
  color:#fff;
}}
.stButton > button[kind="primary"]{{
  background:var(--red);
  border-color:var(--red);
}}
.stButton > button[kind="primary"]:hover{{
  background:var(--red2);
  border-color:var(--red2);
}}

/* inputs */
.stTextArea textarea, div[data-baseweb="select"] > div{{
  background:var(--surface)!important;
  border-color:var(--border)!important;
  border-radius:9px!important;
}}

/* custom */
.topbar{{
  display:flex;align-items:center;justify-content:space-between;
  padding:13px 16px;
  border:1px solid rgba(255,255,255,.08);
  border-radius:12px;
  background:rgba(10,10,12,.72);
  backdrop-filter:blur(12px);
  -webkit-backdrop-filter:blur(12px);
  margin-bottom:10px;
}}
.logo{{
  font-size:1.08rem;font-weight:850;letter-spacing:.02em;color:#fff;
}}
.logo-mark{{color:var(--red);font-size:1.25rem;margin-right:8px}}
.market-status{{
  color:var(--muted);font-size:.78rem;font-weight:650;
}}
.live-dot{{
  display:inline-block;width:7px;height:7px;border-radius:50%;
  background:var(--green);margin-right:7px;
  box-shadow:0 0 9px rgba(22,199,132,.55);
}}
.instrument{{
  padding:24px 2px 12px 2px;
}}
.symbol{{font-size:.78rem;color:var(--muted);font-weight:750;letter-spacing:.09em}}
.asset-title{{font-size:1.1rem;font-weight:700;margin-top:4px}}
.price-line{{display:flex;align-items:flex-end;gap:15px;margin-top:7px}}
.price{{font-size:2.8rem;line-height:1;font-weight:780;letter-spacing:-.04em}}
.unit{{font-size:.9rem;color:var(--muted);padding-bottom:5px}}
.change-up{{color:var(--green);font-weight:750;padding-bottom:5px}}
.change-down{{color:var(--danger);font-weight:750;padding-bottom:5px}}
.updated{{color:var(--muted);font-size:.76rem;margin-top:10px}}
.section-label{{
  font-size:.79rem;color:var(--muted);font-weight:750;
  text-transform:uppercase;letter-spacing:.08em;margin:10px 0 9px 0;
}}
.panel{{
  background:rgba(17,17,19,.90);
  border:1px solid rgba(255,255,255,.09);
  border-radius:12px;
  padding:18px 19px;
  backdrop-filter:blur(10px);
  -webkit-backdrop-filter:blur(10px);
}}
.row{{
  display:flex;justify-content:space-between;gap:18px;
  padding:10px 0;border-bottom:1px solid #222226;
}}
.row:last-child{{border-bottom:0}}
.row-name{{color:var(--muted);font-size:.88rem}}
.row-value{{color:#fff;font-size:.9rem;font-weight:700}}
.insight-head{{
  font-size:1.15rem;font-weight:760;margin-bottom:6px
}}
.subtle{{color:var(--muted);font-size:.84rem;line-height:1.55}}
.redline{{
  height:3px;width:42px;background:var(--red);border-radius:99px;margin:12px 0 16px
}}
[data-testid="stDataFrame"]{{
  border:1px solid var(--border);
  border-radius:10px;
  overflow:hidden;
}}

/* V9.1: UI stays fully opaque above the background */
.topbar, .instrument, .section-label,
[data-testid="stMetric"], [data-testid="stDataFrame"],
[data-testid="stPlotlyChart"], .panel,
.stButton, .stTextArea, [data-baseweb="tab-list"] {{
  position:relative;
  z-index:3;
}}
.logo, .asset-title, .price, .unit, .row-value,
div[data-testid="stMetricValue"], button[data-baseweb="tab"] {{
  opacity:1 !important;
}}
</style>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <img class="gold-fixed-bg" src="data:image/png;base64,{BG_IMAGE}" alt="">
    <div class="gold-bg-overlay"></div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Firebase Authentication gate
# =========================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if not st.session_state.authenticated:
    st.markdown("""
    <div style="max-width:520px;margin:5vh auto 1.25rem auto;text-align:center;">
      <div style="font-size:.78rem;letter-spacing:.18em;color:#e32636;font-weight:800;">SECURE ACCESS</div>
      <div style="font-size:2.25rem;font-weight:850;margin-top:.35rem;">Gold Analytics</div>
      <div style="color:#96969f;margin-top:.35rem;">Sign in or create an account to continue</div>
    </div>
    """, unsafe_allow_html=True)

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            login_email = st.text_input("Email", placeholder="name@example.com")
            login_password = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Login", use_container_width=True)
        if login_submit:
            if not login_email or not login_password:
                st.warning("กรุณากรอก Email และ Password")
            else:
                try:
                    user = login_user(login_email, login_password)
                    st.session_state.authenticated = True
                    st.session_state.user_email = user.get("email", login_email)
                    st.session_state.id_token = user.get("idToken", "")
                    st.rerun()
                except Exception as e:
                    st.error(f"Login ไม่สำเร็จ: {e}")

    with register_tab:
        with st.form("register_form"):
            reg_email = st.text_input("Email", placeholder="name@example.com", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password", help="อย่างน้อย 6 ตัวอักษร")
            reg_confirm = st.text_input("Confirm password", type="password", key="reg_confirm")
            reg_submit = st.form_submit_button("Create account", use_container_width=True)
        if reg_submit:
            if not reg_email or not reg_password:
                st.warning("กรุณากรอก Email และ Password")
            elif reg_password != reg_confirm:
                st.error("Password และ Confirm password ไม่ตรงกัน")
            elif len(reg_password) < 6:
                st.error("Password ต้องมีอย่างน้อย 6 ตัวอักษร")
            else:
                try:
                    user = register_user(reg_email, reg_password)
                    st.session_state.authenticated = True
                    st.session_state.user_email = user.get("email", reg_email)
                    st.session_state.id_token = user.get("idToken", "")
                    st.success("สร้างบัญชีสำเร็จ")
                    st.rerun()
                except Exception as e:
                    st.error(f"สมัครสมาชิกไม่สำเร็จ: {e}")

    st.stop()

@st.cache_data(ttl=300)
def load_spot():
    return get_gold_spot()

@st.cache_data(ttl=3600)
def load_history():
    time.sleep(2)
    return get_gold_history()

def money(v):
    return "N/A" if pd.isna(v) else f"${v:,.2f}"

def pct(v, signed=True):
    if pd.isna(v): return "N/A"
    return f"{v:+.2f}%" if signed else f"{v:.2f}%"

spot = load_spot()
history = load_history()

latest = float(history.iloc[-1]["Gold_Price"])
previous = float(history.iloc[-2]["Gold_Price"])
day_change = (latest / previous - 1) * 100
change_class = "change-up" if day_change >= 0 else "change-down"

st.markdown(f"""
<div class="topbar">
  <div class="logo"><span class="logo-mark">◆</span>GOLD ANALYTICS</div>
  <div class="market-status"><span class="live-dot"></span>XAU/USD · GOLDPRICE.DEV</div>
</div>
""", unsafe_allow_html=True)

user_col, logout_col = st.columns([8, 1.4])
with user_col:
    st.caption(f"Signed in as {st.session_state.user_email}")
with logout_col:
    if st.button("Logout", use_container_width=True):
        for key in ["authenticated", "user_email", "id_token"]:
            st.session_state.pop(key, None)
        st.rerun()

tabs = st.tabs(["Overview", "Market Data", "Analytics", "Market Insights"])

# ---------- OVERVIEW ----------
with tabs[0]:
    st.markdown(f"""
    <div class="instrument">
      <div class="symbol">XAU / USD</div>
      <div class="asset-title">Gold / U.S. Dollar</div>
      <div class="price-line">
        <div class="price">{spot["price"]:,.2f}</div>
        <div class="unit">USD</div>
        <div class="{change_class}">{day_change:+.2f}% daily</div>
      </div>
      <div class="updated">Spot updated {spot["timestamp"]}</div>
    </div>
    """, unsafe_allow_html=True)

    period = st.radio(
        "Range",
        ["7D", "30D"],
        horizontal=True,
        index=1,
        label_visibility="collapsed",
        key="overview_period",
    )
    n = {"7D":7, "30D":30}[period]
    selected = history.tail(n).copy()
    stats, df = calculate_statistics(selected)

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("PERFORMANCE", pct(stats["period_return"]))
    k2.metric("VOLATILITY", pct(stats["daily_volatility"], False))
    k3.metric("PERIOD HIGH", money(stats["max"]))
    k4.metric("PERIOD LOW", money(stats["min"]))

    st.markdown('<div class="section-label">Price performance</div>', unsafe_allow_html=True)

    fig = go.Figure()
    # area-like main price trace
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Gold_Price"],
        mode="lines", name="XAU/USD",
        line=dict(color="#f1f1f3", width=2.2),
        fill="tozeroy",
        fillcolor="rgba(227,38,54,0.07)",
        hovertemplate="%{x|%d %b %Y}<br><b>$%{y:,.2f}</b><extra></extra>",
    ))
    if df["MA7"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["MA7"], mode="lines", name="MA7",
            line=dict(color="#e32636", width=1.5),
        ))
    if df["MA30"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["MA30"], mode="lines", name="MA30",
            line=dict(color="#8d8d96", width=1.25, dash="dot"),
        ))
    ymin = df["Gold_Price"].min()
    yrange = max(df["Gold_Price"].max()-ymin, 1)
    fig.update_yaxes(range=[ymin-yrange*.12, df["Gold_Price"].max()+yrange*.08])
    fig.update_layout(
        height=480,
        paper_bgcolor="#111113",
        plot_bgcolor="#111113",
        font=dict(color="#a2a2aa"),
        margin=dict(l=15,r=15,t=12,b=10),
        hovermode="x unified",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(gridcolor="#252529", zeroline=False, side="right", tickprefix="$"),
        legend=dict(orientation="h", x=0, y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    c1,c2 = st.columns([1,1])
    with c1:
        st.markdown('<div class="section-label">Market statistics</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="panel">
          <div class="row"><span class="row-name">Mean price</span><span class="row-value">{money(stats["mean"])}</span></div>
          <div class="row"><span class="row-name">Median price</span><span class="row-value">{money(stats["median"])}</span></div>
          <div class="row"><span class="row-name">Standard deviation</span><span class="row-value">{money(stats["std"])}</span></div>
          <div class="row"><span class="row-name">Average daily return</span><span class="row-value">{pct(stats["average_daily_return"])}</span></div>
          <div class="row"><span class="row-name">Observations</span><span class="row-value">{len(df):,}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-label">Trend indicators</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="panel">
          <div class="row"><span class="row-name">Latest close</span><span class="row-value">{money(latest)}</span></div>
          <div class="row"><span class="row-name">Previous close</span><span class="row-value">{money(previous)}</span></div>
          <div class="row"><span class="row-name">Latest daily return</span><span class="row-value">{pct(stats["latest_daily_return"])}</span></div>
          <div class="row"><span class="row-name">MA 7</span><span class="row-value">{money(stats["ma7"])}</span></div>
          <div class="row"><span class="row-name">MA 30</span><span class="row-value">{money(stats["ma30"])}</span></div>
        </div>
        """, unsafe_allow_html=True)

# ---------- MARKET DATA ----------
with tabs[1]:
    st.markdown('<div class="section-label">Historical XAU/USD</div>', unsafe_allow_html=True)
    range2 = st.radio("Range",["7D","30D"],horizontal=True,index=1,label_visibility="collapsed",key="data_period")
    n2={"7D":7,"30D":30}[range2]
    show=history.tail(n2).sort_values("Date",ascending=False)
    m1,m2,m3=st.columns(3)
    m1.metric("SPOT",money(spot["price"]))
    m2.metric("LATEST CLOSE",money(latest),pct(day_change))
    m3.metric("PREVIOUS CLOSE",money(previous))
    st.dataframe(show,hide_index=True,use_container_width=True,height=610)

# ---------- ANALYTICS ----------
with tabs[2]:
    period3=st.radio("Range",["7D","30D"],horizontal=True,index=1,label_visibility="collapsed",key="analytics_period")
    n3={"7D":7,"30D":30}[period3]
    stats3,df3=calculate_statistics(history.tail(n3).copy())

    st.markdown('<div class="section-label">Statistical analytics</div>',unsafe_allow_html=True)
    a1,a2,a3,a4=st.columns(4)
    a1.metric("MEAN",money(stats3["mean"]))
    a2.metric("STANDARD DEVIATION",money(stats3["std"]))
    a3.metric("PERIOD RETURN",pct(stats3["period_return"]))
    a4.metric("DAILY VOLATILITY",pct(stats3["daily_volatility"],False))

    rdf=df3.dropna(subset=["Daily_Return"])
    rfig=go.Figure(go.Bar(
        x=rdf["Date"],y=rdf["Daily_Return"],
        marker_color=[
            "#16c784" if x>=0 else "#ea3943"
            for x in rdf["Daily_Return"]
        ],
        hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}%<extra></extra>"
    ))
    rfig.add_hline(y=0,line_color="#55555d",line_width=1)
    rfig.update_layout(
        height=450,paper_bgcolor="#111113",plot_bgcolor="#111113",
        font=dict(color="#a2a2aa"),margin=dict(l=15,r=15,t=15,b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#252529",ticksuffix="%")
    )
    st.plotly_chart(rfig,use_container_width=True,config={"displayModeBar":False})

# ---------- INSIGHTS ----------
with tabs[3]:
    period4=st.radio("Range",["7D","30D"],horizontal=True,index=1,label_visibility="collapsed",key="insight_period")
    n4={"7D":7,"30D":30}[period4]
    stats4,df4=calculate_statistics(history.tail(n4).copy())

    left,right=st.columns([1.55,1])
    with left:
        st.markdown("""
        <div class="section-label">Market intelligence</div>
        <div class="panel">
          <div class="insight-head">Analyze the current XAU/USD dataset</div>
          <div class="redline"></div>
          <div class="subtle">Ask about trend, return, volatility or moving averages. The analysis uses the selected market statistics rather than raw thousands of rows.</div>
        </div>
        """,unsafe_allow_html=True)
        st.write("")
        # Prompt presets use session_state so the selected prompt is visibly
        # inserted into the text area before the user sends it to Gemini.
        if "market_question" not in st.session_state:
            st.session_state["market_question"] = ""

        def set_market_question(prompt):
            st.session_state["market_question"] = prompt

        question=st.text_area(
            "Question",
            key="market_question",
            placeholder="เช่น วิเคราะห์แนวโน้มของราคาทอง โดยพิจารณา Return, Volatility, MA7 และ MA30",
            height=130,label_visibility="collapsed"
        )
        q1,q2,q3=st.columns(3)
        q1.button(
            "Trend", use_container_width=True,
            on_click=set_market_question,
            args=("วิเคราะห์แนวโน้มราคาทอง XAU/USD ในช่วงเวลาที่เลือก โดยพิจารณา Period Return, MA7 และ MA30 พร้อมอธิบายแนวโน้มระยะสั้นและระยะกลาง",)
        )
        q2.button(
            "Volatility", use_container_width=True,
            on_click=set_market_question,
            args=("วิเคราะห์ความผันผวนของราคาทอง XAU/USD ในช่วงเวลาที่เลือก โดยพิจารณา Daily Return และ Daily Volatility พร้อมอธิบายว่าความผันผวนอยู่ในลักษณะใด",)
        )
        q3.button(
            "Full summary", use_container_width=True,
            on_click=set_market_question,
            args=("สรุปภาพรวมราคาทอง XAU/USD ในช่วงเวลาที่เลือกจากสถิติทั้งหมดที่มี ได้แก่ ราคาปัจจุบัน Period Return, Daily Volatility, MA7 และ MA30 พร้อมสรุปประเด็นสำคัญที่ควรสังเกต",)
        )
        if st.button("Analyze market",type="primary",use_container_width=True):
            if not question.strip():
                st.warning("กรุณาระบุคำถาม")
            else:
                try:
                    with st.spinner("Analyzing market data..."):
                        st.session_state["market_insight"]=ask_gold_ai(
                            question,period4,spot,stats4,df4
                        )
                except Exception:
                    st.error("Market analysis service is temporarily unavailable. Please try again shortly.")

    with right:
        st.markdown('<div class="section-label">Current snapshot</div>',unsafe_allow_html=True)
        st.markdown(f"""
        <div class="panel">
          <div class="row"><span class="row-name">Spot</span><span class="row-value">{money(spot["price"])}</span></div>
          <div class="row"><span class="row-name">Period return</span><span class="row-value">{pct(stats4["period_return"])}</span></div>
          <div class="row"><span class="row-name">Volatility</span><span class="row-value">{pct(stats4["daily_volatility"],False)}</span></div>
          <div class="row"><span class="row-name">MA 7</span><span class="row-value">{money(stats4["ma7"])}</span></div>
          <div class="row"><span class="row-name">MA 30</span><span class="row-value">{money(stats4["ma30"])}</span></div>
        </div>
        """,unsafe_allow_html=True)

    if st.session_state.get("market_insight"):
        st.markdown('<div class="section-label">Analysis</div>',unsafe_allow_html=True)
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown(st.session_state["market_insight"])
        st.markdown('</div>',unsafe_allow_html=True)
