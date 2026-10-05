import base64
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai_service import ask_sales_ai, build_context
from firebase_auth import login_user, register_user
from ml_service import (FEATURE_LABELS, regression_forecast, timeseries_forecast,
                        train_regression)
from sales_service import (category_summary, clean_sales, daily_sales,
                           generate_sample, overall_kpis, product_summary)

st.set_page_config(page_title="Sales Analytics AI", page_icon="S", layout="wide",
                   initial_sidebar_state="collapsed")

# =========================================================
# Theme (เหมือนเว็บราคาทอง)
# =========================================================
st.markdown("""
<style>
:root{--bg:#09090b;--surface:#111113;--surface2:#17171a;--border:#29292e;
  --red:#e32636;--red2:#b91c2b;--green:#16c784;--danger:#ea3943;--text:#f7f7f8;--muted:#96969f;}
html, body, [class*="css"] { font-family: Inter, Arial, sans-serif; }
.stApp { color:var(--text); background:#09090b; }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background:transparent !important; }
[data-testid="stHeader"] { background:transparent; height:0px; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display:none; }
#MainMenu, footer { visibility:hidden; }
.block-container{ max-width:1480px; padding-top:1.1rem; padding-bottom:3rem; position:relative; z-index:2; }
h1,h2,h3{letter-spacing:-.025em}
button[data-baseweb="tab"]{ font-weight:650!important; color:#aaaab2!important; }
button[data-baseweb="tab"][aria-selected="true"]{ color:#fff!important; }
div[data-baseweb="tab-highlight"]{ background-color:var(--red)!important; }
div[data-testid="stMetric"]{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:15px 17px; }
div[data-testid="stMetricLabel"]{color:var(--muted)}
div[data-testid="stMetricValue"]{font-size:1.42rem}
.stButton > button{ border-radius:8px; min-height:40px; border:1px solid var(--border); background:var(--surface2); color:#fff; font-weight:650; }
.stButton > button:hover{ border-color:var(--red); color:#fff; }
.stButton > button[kind="primary"]{ background:var(--red); border-color:var(--red); }
.stButton > button[kind="primary"]:hover{ background:var(--red2); border-color:var(--red2); }
.stTextArea textarea, div[data-baseweb="select"] > div{ background:var(--surface)!important; border-color:var(--border)!important; border-radius:9px!important; }
[data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:10px; overflow:hidden; }
.topbar{ display:flex;align-items:center;justify-content:space-between; padding:13px 16px;
  border:1px solid rgba(255,255,255,.08); border-radius:12px; background:rgba(10,10,12,.72); margin-bottom:10px; }
.logo{ font-size:1.08rem;font-weight:850;letter-spacing:.02em;color:#fff; }
.logo-mark{color:var(--red);font-size:1.25rem;margin-right:8px}
.market-status{ color:var(--muted);font-size:.78rem;font-weight:650; }
.live-dot{ display:inline-block;width:7px;height:7px;border-radius:50%; background:var(--green);margin-right:7px; box-shadow:0 0 9px rgba(22,199,132,.55); }
.section-label{ font-size:.79rem;color:var(--muted);font-weight:750; text-transform:uppercase;letter-spacing:.08em;margin:14px 0 9px 0; }
.panel{ background:rgba(17,17,19,.90); border:1px solid rgba(255,255,255,.09); border-radius:12px; padding:18px 19px; }
.insight-head{ font-size:1.15rem;font-weight:760;margin-bottom:6px }
.subtle{color:var(--muted);font-size:.84rem;line-height:1.55}
.redline{ height:3px;width:42px;background:var(--red);border-radius:99px;margin:12px 0 16px }
</style>
""", unsafe_allow_html=True)

# พื้นหลังรูปภาพ (ไม่บังคับ: ถ้ามี assets/background.png จะแสดง)
_bg = Path(__file__).parent / "assets" / "background.png"
if _bg.exists():
    b64 = base64.b64encode(_bg.read_bytes()).decode()
    st.markdown(f"""
    <style>
    .bg-img{{position:fixed;inset:0;width:100vw;height:100vh;object-fit:cover;z-index:-2;opacity:.45;pointer-events:none}}
    .bg-ov{{position:fixed;inset:0;z-index:-1;pointer-events:none;background:linear-gradient(180deg,rgba(9,9,11,.4),rgba(9,9,11,.95))}}
    </style>
    <img class="bg-img" src="data:image/png;base64,{b64}" alt=""><div class="bg-ov"></div>
    """, unsafe_allow_html=True)

# =========================================================
# Firebase login (โค้ดเดิม)
# =========================================================
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("user_email", "")

if not st.session_state.authenticated:
    st.markdown("""
    <div style="max-width:520px;margin:5vh auto 1.25rem auto;text-align:center;">
      <div style="font-size:.78rem;letter-spacing:.18em;color:#e32636;font-weight:800;">SECURE ACCESS</div>
      <div style="font-size:2.25rem;font-weight:850;margin-top:.35rem;">Sales Analytics AI</div>
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
                    st.rerun()
                except Exception as e:
                    st.error(f"สมัครสมาชิกไม่สำเร็จ: {e}")
    st.stop()


# =========================================================
# Helpers
# =========================================================
def money(v):
    return "N/A" if pd.isna(v) else f"{v:,.0f}"


def pct(v, signed=True):
    if pd.isna(v):
        return "N/A"
    return f"{v:+.1f}%" if signed else f"{v:.1f}%"


def style(fig, h=420):
    fig.update_layout(height=h, paper_bgcolor="#111113", plot_bgcolor="#111113",
                      font=dict(color="#a2a2aa"), margin=dict(l=15, r=15, t=15, b=10),
                      hovermode="x unified", legend=dict(orientation="h", x=0, y=1.1),
                      xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#252529", zeroline=False))
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def label(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


@st.cache_data
def load_sample():
    return generate_sample()


@st.cache_data
def load_models(daily):
    return train_regression(daily)


@st.cache_data
def load_ts(daily, horizon):
    return timeseries_forecast(daily, horizon)


# =========================================================
# Header + data source
# =========================================================
st.markdown("""
<div class="topbar">
  <div class="logo"><span class="logo-mark">◆</span>SALES ANALYTICS AI</div>
  <div class="market-status"><span class="live-dot"></span>REGRESSION · TIME SERIES · GENERATIVE AI</div>
</div>
""", unsafe_allow_html=True)

user_col, logout_col = st.columns([8, 1.4])
with user_col:
    st.caption(f"Signed in as {st.session_state.user_email}")
with logout_col:
    if st.button("Logout", use_container_width=True):
        for key in ["authenticated", "user_email", "id_token", "sales_insight"]:
            st.session_state.pop(key, None)
        st.rerun()

with st.expander("Data source", expanded=False):
    st.caption("ไฟล์ CSV/Excel ต้องมีคอลัมน์ Date, Product, Quantity, Price "
               "(ไม่บังคับ: Category, Promotion = 0/1, Sales)")
    upload = st.file_uploader("Upload sales file", type=["csv", "xlsx"])

raw, source = clean_sales(load_sample()), "ข้อมูลตัวอย่าง (จำลอง)"
if upload is not None:
    try:
        file_df = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
        raw, source = clean_sales(file_df), upload.name
    except Exception as e:
        st.error(f"อ่านไฟล์ไม่ได้: {e} — กำลังใช้ข้อมูลตัวอย่างแทน")
st.caption(f"แหล่งข้อมูล: {source}")

df = raw
daily = daily_sales(df)
kpis = overall_kpis(daily)
products = product_summary(df)
reg = load_models(daily)

tabs = st.tabs(["Overview", "Factors", "Products", "Forecast", "AI Insights"])

# ---------- OVERVIEW ----------
with tabs[0]:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("TOTAL SALES", money(kpis["total_sales"]))
    k2.metric("AVG / DAY", money(kpis["avg_daily"]))
    k3.metric("LAST 30 DAYS", money(kpis["last30"]), pct(kpis["growth_30d"]))
    k4.metric("TOTAL UNITS", money(kpis["total_qty"]))

    label("Daily sales trend")
    d = daily.copy()
    d["MA7"] = d["Sales"].rolling(7).mean()
    d["MA30"] = d["Sales"].rolling(30).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Sales"], name="Sales", line=dict(color="#555560", width=1)))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["MA7"], name="MA7", line=dict(color="#e32636", width=1.6)))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["MA30"], name="MA30", line=dict(color="#f1f1f3", width=2)))
    show(style(fig, 440))

    c1, c2 = st.columns(2)
    with c1:
        label("Monthly sales")
        m = d.set_index("Date")["Sales"].resample("MS").sum().reset_index()
        show(style(go.Figure(go.Bar(x=m["Date"], y=m["Sales"], marker_color="#e32636")), 340))
    with c2:
        label("Average sales by day of week")
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        w = d.groupby(d["Date"].dt.dayofweek)["Sales"].mean()
        show(style(go.Figure(go.Bar(x=names, y=w.values, marker_color="#f1f1f3")), 340))

# ---------- FACTORS ----------
with tabs[1]:
    if reg is None:
        st.info("ข้อมูลน้อยเกินไปสำหรับวิเคราะห์ปัจจัย")
    else:
        c1, c2 = st.columns(2)
        with c1:
            label("Feature importance (Random Forest)")
            imp = reg["importance"].sort_values()
            fig = go.Figure(go.Bar(x=imp.values, y=[FEATURE_LABELS[i] for i in imp.index],
                                   orientation="h", marker_color="#e32636"))
            fig = style(fig, 420)
            fig.update_layout(hovermode="closest", xaxis=dict(tickformat=".0%", gridcolor="#252529"))
            show(fig)
        with c2:
            label("Correlation with daily sales")
            corr = reg["corr"].sort_values()
            fig = go.Figure(go.Bar(x=corr.values, y=[FEATURE_LABELS[i] for i in corr.index], orientation="h",
                                   marker_color=["#16c784" if v >= 0 else "#ea3943" for v in corr.values]))
            fig = style(fig, 420)
            fig.update_layout(hovermode="closest", xaxis=dict(gridcolor="#252529"))
            show(fig)

        p = daily.groupby("Promotion")["Sales"].mean()
        if 0 in p.index and 1 in p.index and p[0] > 0:
            a, b, c = st.columns(3)
            a.metric("AVG SALES (NORMAL DAY)", money(p[0]))
            b.metric("AVG SALES (PROMO DAY)", money(p[1]))
            c.metric("PROMO UPLIFT", pct((p[1] / p[0] - 1) * 100))
        st.caption("Correlation ไม่ได้แปลว่าเป็นเหตุและผล ควรใช้ประกอบกับความรู้ของธุรกิจ")

# ---------- PRODUCTS ----------
with tabs[2]:
    label("Pareto: sales by product")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=products["Product"], y=products["Sales"], name="Sales", marker_color="#e32636"))
    fig.add_trace(go.Scatter(x=products["Product"], y=products["CumShare"], name="Cumulative %",
                             yaxis="y2", line=dict(color="#f1f1f3", width=2)))
    fig = style(fig, 420)
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105], ticksuffix="%", showgrid=False))
    show(fig)

    c1, c2 = st.columns([1.6, 1])
    with c1:
        label("Product ranking (ABC + 30-day growth)")
        t = products[["Product", "Category", "Sales", "Share", "ABC", "Growth30d"]].rename(
            columns={"Share": "Share %", "Growth30d": "Growth 30d %"})
        st.dataframe(t.round(1), hide_index=True, use_container_width=True, height=360)
    with c2:
        label("Sales by category")
        cat = category_summary(df)
        fig = go.Figure(go.Bar(x=cat["Category"], y=cat["Sales"], marker_color="#f1f1f3"))
        show(style(fig, 360))

    rising = products.dropna(subset=["Growth30d"])
    rising = rising[rising["Growth30d"] > 0].sort_values("Growth30d", ascending=False).head(3)
    if len(rising):
        st.success("สินค้าที่มีศักยภาพ (ยอดโตเร็วสุด): " +
                   ", ".join(f"{r.Product} ({r.Growth30d:+.1f}%)" for r in rising.itertuples()))

# ---------- FORECAST ----------
with tabs[3]:
    horizon = st.slider("Forecast horizon (days)", 7, 60, 30)
    ts_fc, backtest = None, None
    try:
        ts_fc, backtest = load_ts(daily, horizon)
    except Exception as e:
        st.warning(f"Time series forecast ไม่สำเร็จ: {e}")

    hist = daily.tail(90)
    label("Future sales forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Sales"], name="Actual", line=dict(color="#f1f1f3", width=1.8)))
    if ts_fc is not None:
        fig.add_trace(go.Scatter(x=ts_fc["Date"], y=ts_fc["Upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ts_fc["Date"], y=ts_fc["Lower"], fill="tonexty", fillcolor="rgba(227,38,54,.15)",
                                 line=dict(width=0), name="95% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ts_fc["Date"], y=ts_fc["Forecast"], name="Holt-Winters",
                                 line=dict(color="#e32636", width=2.2)))
    if reg is not None:
        rf = regression_forecast(reg["model"], daily, horizon)
        fig.add_trace(go.Scatter(x=rf["Date"], y=rf["Forecast"], name=f"Regression ({reg['best']})",
                                 line=dict(color="#16c784", width=2, dash="dot")))
    show(style(fig, 460))

    if ts_fc is not None:
        a, b, c = st.columns(3)
        a.metric(f"TOTAL NEXT {horizon} DAYS", money(ts_fc["Forecast"].sum()))
        b.metric("AVG / DAY", money(ts_fc["Forecast"].mean()))
        if backtest:
            c.metric("BACKTEST MAE (28D)", money(backtest["MAE"]), f"baseline {money(backtest['Naive_MAE'])}", delta_color="off")

    if reg is not None:
        label("Regression model evaluation (time-based test set)")
        st.dataframe(reg["metrics"].round(3), hide_index=True, use_container_width=True)
        t = reg["test"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t["Date"], y=t["Actual"], name="Actual", line=dict(color="#f1f1f3", width=1.8)))
        fig.add_trace(go.Scatter(x=t["Date"], y=t[reg["best"]], name=reg["best"], line=dict(color="#e32636", width=1.8)))
        show(style(fig, 340))
        st.caption("Regression ประเมินแบบทำนายล่วงหน้า 1 วัน ส่วนเส้นประในกราฟอนาคตเป็นการทำนายต่อเนื่อง "
                   "(ใช้ผลทำนายเป็น lag) จึงคลาดเคลื่อนสะสมได้ และสมมติว่าไม่มีโปรโมชัน")

# ---------- AI INSIGHTS ----------
with tabs[4]:
    ts_ctx, bt_ctx = None, None
    try:
        ts_ctx, bt_ctx = load_ts(daily, 30)
    except Exception:
        pass
    context = build_context(kpis, products, daily, reg, ts_ctx, bt_ctx)

    left, right = st.columns([1.55, 1])
    with left:
        st.markdown("""
        <div class="section-label">Business intelligence</div>
        <div class="panel">
          <div class="insight-head">สรุปผลและข้อเสนอแนะจาก Generative AI</div>
          <div class="redline"></div>
          <div class="subtle">AI จะได้รับเฉพาะตัวเลขสรุปที่คำนวณจากข้อมูลจริง ไม่ได้รับข้อมูลดิบทั้งตาราง</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.session_state.setdefault("sales_question", "")

        def set_q(text):
            st.session_state["sales_question"] = text

        question = st.text_area("Question", key="sales_question", height=130, label_visibility="collapsed",
                                placeholder="เช่น สินค้าไหนควรเน้นเพิ่มสต็อกหรือทำโปรโมชันในเดือนหน้า")
        q1, q2, q3 = st.columns(3)
        q1.button("Full summary", use_container_width=True, on_click=set_q,
                  args=("สรุปภาพรวมยอดขาย แนวโน้ม ปัจจัยที่ส่งผล และสินค้าที่ควรให้ความสำคัญ",))
        q2.button("Why sales change", use_container_width=True, on_click=set_q,
                  args=("วิเคราะห์สาเหตุที่ยอดขายเปลี่ยนแปลงในช่วง 30 วันล่าสุด โดยพิจารณาปัจจัยและสินค้าแต่ละตัว",))
        q3.button("Action plan", use_container_width=True, on_click=set_q,
                  args=("เสนอแผนปฏิบัติ 5 ข้อเพื่อเพิ่มยอดขาย โดยอิงจากสินค้า ปัจจัย และผลพยากรณ์",))
        if st.button("Analyze sales", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("กรุณาระบุคำถาม")
            else:
                try:
                    with st.spinner("Analyzing sales data..."):
                        st.session_state["sales_insight"] = ask_sales_ai(question, context)
                except Exception as e:
                    st.error(f"ใช้งาน AI ไม่ได้ในขณะนี้: {e}")
    with right:
        label("Data sent to AI")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.text(context)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("sales_insight"):
        label("Analysis")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(st.session_state["sales_insight"])
        st.markdown("</div>", unsafe_allow_html=True)
