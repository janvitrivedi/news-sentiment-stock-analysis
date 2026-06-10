import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI News Impact Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

COMPANY_NAMES = {
    "ADBE":"Adobe Inc","ADI":"Analog Devices","ADP":"Automatic Data Processing",
    "ALGN":"Align Technology","AMAT":"Applied Materials","ASML":"ASML Holding",
    "AVGO":"Broadcom Inc","BIDU":"Baidu Inc","BIIB":"Biogen Inc",
    "CMCSA":"Comcast Corp","CSCO":"Cisco Systems","CTAS":"Cintas Corp",
    "CTSH":"Cognizant Technology","DLTR":"Dollar Tree","EA":"Electronic Arts",
    "EBAY":"eBay Inc","ENPH":"Enphase Energy","FAST":"Fastenal Co",
    "FTNT":"Fortinet Inc","GILD":"Gilead Sciences","GOOG":"Alphabet (GOOG)",
    "GOOGL":"Alphabet (GOOGL)","ILMN":"Illumina Inc","INTU":"Intuit Inc",
    "JD":"JD.com Inc","KLAC":"KLA Corp","LRCX":"Lam Research",
    "MCHP":"Microchip Technology","MNST":"Monster Beverage","MRVL":"Marvell Technology",
    "MU":"Micron Technology","NFLX":"Netflix Inc","NTES":"NetEase Inc",
    "NVDA":"NVIDIA Corp","NXPI":"NXP Semiconductors","ORCL":"Oracle Corp",
    "ORLY":"O'Reilly Automotive","PANW":"Palo Alto Networks","PAYX":"Paychex Inc",
    "QCOM":"Qualcomm Inc","REGN":"Regeneron Pharmaceuticals","SBUX":"Starbucks Corp",
    "SIRI":"SiriusXM Holdings","TSLA":"Tesla Inc","TXN":"Texas Instruments",
    "VRTX":"Vertex Pharmaceuticals","WDAY":"Workday Inc",
    "AAPL":"Apple Inc","MSFT":"Microsoft Corp","AMZN":"Amazon.com",
    "META":"Meta Platforms","COST":"Costco Wholesale",
}

st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 15px !important; }
    .stApp { background-color: #f0f4f8; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1f36 0%, #2d3561 100%); }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * { color: #1a202c !important; background-color: white !important; }
    [data-testid="stSidebar"] .stSelectbox input { color: #1a202c !important; }
    [data-testid="stSidebar"] label { font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; color: #a0aec0 !important; font-weight: 600; }

    .hero { background: linear-gradient(135deg, #1a1f36 0%, #2d3561 50%, #1a6b8a 100%); border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.2rem; color: white; }
    .hero-title { font-size: 2.2rem !important; font-weight: 900; color: white; margin: 0; letter-spacing: -0.02em; }
    .hero-sub { font-size: 0.95rem !important; color: #a0c4ff; margin: 0.3rem 0 0.7rem 0; }
    .hero-badge { display: inline-block; background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.2); border-radius: 20px; padding: 0.2rem 0.75rem; font-size: 0.75rem !important; color: white; margin-right: 0.4rem; margin-top: 0.25rem; }
    .hero-ticker { font-size: 2.5rem !important; font-weight: 900; color: white; line-height: 1; }
    .hero-company { font-size: 0.8rem; color: #a0c4ff; margin-top: 0.2rem; }

    .kpi { background: white; border-radius: 11px; padding: 1rem 1.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-left: 4px solid #4361ee; }
    .kpi-blue { border-left-color: #4361ee; }
    .kpi-green { border-left-color: #06d6a0; }
    .kpi-red { border-left-color: #ef476f; }
    .kpi-orange { border-left-color: #ffd166; }
    .kpi-purple { border-left-color: #7209b7; }
    .kpi-label { font-size: 0.68rem !important; text-transform: uppercase; letter-spacing: 0.1em; color: #718096; font-weight: 700; margin-bottom: 0.2rem; }
    .kpi-value { font-size: 1.7rem !important; font-weight: 900; color: #1a202c; line-height: 1.1; }
    .kpi-sub { font-size: 0.72rem !important; color: #a0aec0; margin-top: 0.1rem; }
    .kpi-trend { font-size: 0.7rem !important; font-weight: 600; margin-top: 0.15rem; }

    .sec-hdr { font-size: 1.05rem !important; font-weight: 800; color: #1a202c; padding-bottom: 0.4rem; border-bottom: 3px solid #4361ee; margin: 1.4rem 0 0.8rem 0; display: block; }

    .card { background: white; border-radius: 11px; padding: 1rem 1.2rem; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }
    .card-title { font-size: 0.82rem !important; font-weight: 700; color: #1a202c; margin-bottom: 0.5rem; }
    .dr { display:flex; justify-content:space-between; padding:0.32rem 0; border-bottom:1px solid #f1f5f9; font-size:0.75rem; }
    .dr:last-child { border-bottom:none; }
    .dl { color:#64748b; } .dv { font-weight:700; color:#1a202c; }

    .find { background:white; border-radius:11px; padding:1.1rem 1.2rem; box-shadow:0 2px 6px rgba(0,0,0,0.05); height:100%; }
    .fn { font-size:1.5rem !important; font-weight:900; color:#4361ee; line-height:1; }
    .ft { font-size:0.8rem !important; color:#4a5568; line-height:1.55; margin-top:0.35rem; }

    .nc { background:white; border-radius:9px; padding:0.7rem 0.9rem; margin-bottom:0.4rem; border-left:4px solid #e2e8f0; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
    .nc-p { border-left-color:#06d6a0; } .nc-n { border-left-color:#ef476f; } .nc-z { border-left-color:#ffd166; }
    .nh { font-size:0.8rem !important; font-weight:600; color:#2d3748; line-height:1.35; }
    .nm { font-size:0.68rem !important; color:#a0aec0; margin-top:0.2rem; }
    .bp { background:#c6f6d5;color:#276749;border-radius:9px;padding:0.1rem 0.4rem;font-size:0.65rem;font-weight:700; }
    .bn { background:#fed7d7;color:#9b2335;border-radius:9px;padding:0.1rem 0.4rem;font-size:0.65rem;font-weight:700; }
    .bz { background:#fefcbf;color:#7b6514;border-radius:9px;padding:0.1rem 0.4rem;font-size:0.65rem;font-weight:700; }

    .pred-up { background:#ecfdf5; border-radius:7px; padding:0.5rem 0.75rem; margin-bottom:0.3rem; border-left:3px solid #06d6a0; display:flex; justify-content:space-between; align-items:center; }
    .pred-dn { background:#fef2f2; border-radius:7px; padding:0.5rem 0.75rem; margin-bottom:0.3rem; border-left:3px solid #ef476f; display:flex; justify-content:space-between; align-items:center; }

    .footer { background:#1a1f36; border-radius:12px; padding:1.1rem 1.4rem; color:#a0aec0; font-size:0.75rem; margin-top:1.8rem; text-align:center; line-height:2; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    merged  = pd.read_csv("merged_dataset.csv")
    sent    = pd.read_csv("kaggle_sentiment_scores.csv")
    corr    = pd.read_csv("correlation_results.csv")
    ml      = pd.read_csv("ml_results.csv")
    stock   = pd.read_csv("stock_prices_kaggle.csv")
    event   = pd.read_csv("event_study_results.csv")
    anomaly = pd.read_csv("anomaly_results.csv")
    sens    = pd.read_csv("ticker_sensitivity.csv")
    recent  = pd.read_csv("sentiment_scores.csv")
    pred    = pd.read_csv("predictions_2026.csv")
    return merged, sent, corr, ml, stock, event, anomaly, sens, recent, pred

try:
    merged_df,sent_df,corr_df,ml_df,stock_df,event_df,anomaly_df,sens_df,recent_df,pred_df = load_data()
    merged_df["date"] = pd.to_datetime(merged_df["date"])
    stock_df["date"]  = pd.to_datetime(stock_df["date"])
    sent_df["date"]   = pd.to_datetime(sent_df["date"], errors="coerce")
    recent_df["date"] = pd.to_datetime(recent_df["date"], utc=True).dt.tz_localize(None)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

CT = dict(paper_bgcolor="white", plot_bgcolor="white",
          font=dict(size=11), margin=dict(l=8,r=8,t=40,b=8),
          title_font=dict(size=12, color="#1a202c"))

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.1rem 0 0.7rem;'>
        <div style='font-size:2rem;'>📈</div>
        <div style='font-size:1rem;font-weight:900;color:white;line-height:1.2;'>AI News Impact Analyzer</div>
        <div style='font-size:0.7rem;color:#a0aec0;margin-top:0.25rem;'>DAMO 699 · Group 3</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    tickers = sorted(merged_df["ticker"].dropna().unique())
    sel = st.selectbox("SELECT STOCK", tickers,
                       index=tickers.index("NVDA") if "NVDA" in tickers else 0,
                       format_func=lambda x: f"{x} — {COMPANY_NAMES.get(x,x)}")
    st.markdown("---")
    model = st.radio("SENTIMENT MODEL", ["TextBlob","VADER"], index=1)
    st.markdown("---")
    min_yr, max_yr = 2009, 2020
    date_range = st.slider("DATE RANGE", min_yr, max_yr, (min_yr, max_yr), step=1)
    st.markdown("---")
    show_raw = st.checkbox("Show Raw Data", value=False)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem;color:#a0aec0;line-height:2.2;'>
        <div style='color:white;font-size:0.8rem;font-weight:800;margin-bottom:0.2rem;'>Training Data</div>
        📰 68,043 headlines · Kaggle<br>
        📅 2009–2019 · 47 NASDAQ stocks<br><br>
        <div style='color:white;font-size:0.8rem;font-weight:800;margin-bottom:0.2rem;'>Testing Data</div>
        📋 2020 · held-out · unseen by model<br><br>
        <div style='color:white;font-size:0.8rem;font-weight:800;margin-bottom:0.2rem;'>Validation Data</div>
        🌐 979 headlines · NewsAPI<br>
        📅 May 2026 · real-world<br><br>
        <div style='color:white;font-size:0.8rem;font-weight:800;margin-bottom:0.2rem;'>NLP Models</div>
        🔵 TextBlob &nbsp;·&nbsp; 🟢 VADER<br><br>
        <div style='color:white;font-size:0.8rem;font-weight:800;margin-bottom:0.2rem;'>ML Models</div>
        ⚡ Logistic Regression<br>
        🌲 Random Forest
    </div>""", unsafe_allow_html=True)

# ── COMPUTED VALUES ───────────────────────────────────────
t_merged  = merged_df[merged_df["ticker"]==sel].copy()
t_stock   = stock_df[stock_df["ticker"]==sel].copy()
t_sent    = sent_df[sent_df["ticker"]==sel].copy()
t_corr    = corr_df[corr_df["ticker"]==sel]
t_anomaly = anomaly_df[anomaly_df["ticker"]==sel].copy()
t_recent  = recent_df[recent_df["ticker"]==sel].copy()
t_pred    = pred_df[pred_df["ticker"]==sel].copy()

yr_start = pd.Timestamp(f"{date_range[0]}-01-01")
yr_end   = pd.Timestamp(f"{date_range[1]}-12-31")
t_merged_f = t_merged[(t_merged["date"]>=yr_start)&(t_merged["date"]<=yr_end)]
t_stock_f  = t_stock[(t_stock["date"]>=yr_start)&(t_stock["date"]<=yr_end)]

scol = "avg_textblob_score" if model=="TextBlob" else "avg_vader_score"
ccol = "textblob_correlation" if model=="TextBlob" else "vader_correlation"

corr_val = round(t_corr[ccol].values[0],3) if not t_corr.empty else 0
avg_sent = round(t_merged[scol].mean(),3) if not t_merged.empty else 0
matched  = len(t_merged)
articles = len(t_sent)
pval     = round(t_corr["vader_pvalue"].values[0],4) if not t_corr.empty else 1
company  = COMPANY_NAMES.get(sel, sel)
sig_n    = len(corr_df[corr_df["vader_pvalue"]<0.05])

lr = ml_df[ml_df["model"]=="Logistic Regression"].iloc[0]
rf = ml_df[ml_df["model"]=="Random Forest"].iloc[0]

if not t_anomaly.empty and "is_anomaly" in t_anomaly.columns:
    anom_rate = round(t_anomaly["is_anomaly"].sum()/len(t_anomaly)*100,1)
else:
    anom_rate = 0

overall_anom = round(anomaly_df["is_anomaly"].sum()/len(anomaly_df)*100,2)
pos0 = event_df[(event_df["event_type"]=="Positive")&(event_df["lag_day"]==0)]["avg_return"].values[0]
neg0 = event_df[(event_df["event_type"]=="Negative")&(event_df["lag_day"]==0)]["avg_return"].values[0]
news_min = str(t_sent["date"].min())[:10] if not t_sent.empty else "N/A"
news_max = str(t_sent["date"].max())[:10] if not t_sent.empty else "N/A"

# ── HERO ─────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
        <div>
            <div class="hero-title">📈 AI News Impact Analyzer</div>
            <div class="hero-sub">How financial news sentiment moves NASDAQ-100 stock prices</div>
            <div style="margin-top:0.5rem;">
                <span class="hero-badge">📰 68,043 Headlines</span>
                <span class="hero-badge">📊 47 NASDAQ Stocks</span>
                <span class="hero-badge">🎓 Train: 2009–2019</span>
                <span class="hero-badge">🧪 Test: 2020</span>
                <span class="hero-badge">✅ Validation: May 2026</span>
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:0.68rem;color:#a0c4ff;text-transform:uppercase;letter-spacing:0.08em;">Currently Viewing</div>
            <div class="hero-ticker">{sel}</div>
            <div class="hero-company">{company}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI CARDS ────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)

with k1:
    st.markdown(f"""<div class="kpi kpi-blue">
        <div class="kpi-label">News Articles</div>
        <div class="kpi-value">{articles:,}</div>
        <div class="kpi-sub">training headlines</div>
        <div class="kpi-trend" style="color:#718096;">{news_min[:7]} – {news_max[:7]}</div>
    </div>""", unsafe_allow_html=True)

with k2:
    c = "kpi-green" if avg_sent>0.05 else "kpi-red" if avg_sent<-0.05 else "kpi-orange"
    t = "Positive sentiment" if avg_sent>0.05 else "Negative sentiment" if avg_sent<-0.05 else "Neutral sentiment"
    st.markdown(f"""<div class="kpi {c}">
        <div class="kpi-label">{model} Score</div>
        <div class="kpi-value">{avg_sent}</div>
        <div class="kpi-sub">avg sentiment score</div>
        <div class="kpi-trend">{t}</div>
    </div>""", unsafe_allow_html=True)

with k3:
    c = "kpi-green" if corr_val>0.15 else "kpi-red" if corr_val<0 else "kpi-orange"
    t = "Strong signal" if abs(corr_val)>0.3 else "Moderate signal" if abs(corr_val)>0.15 else "Weak signal"
    st.markdown(f"""<div class="kpi {c}">
        <div class="kpi-label">Correlation (r)</div>
        <div class="kpi-value">{corr_val}</div>
        <div class="kpi-sub">sentiment vs returns</div>
        <div class="kpi-trend">{t} · p={pval}</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""<div class="kpi kpi-orange">
        <div class="kpi-label">Anomaly Rate</div>
        <div class="kpi-value">{anom_rate}%</div>
        <div class="kpi-sub">sentiment-price mismatch</div>
        <div class="kpi-trend" style="color:#06d6a0;"> {round(100-anom_rate,1)}% correctly predicted</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""<div class="kpi kpi-purple">
        <div class="kpi-label">Matched Days</div>
        <div class="kpi-value">{matched:,}</div>
        <div class="kpi-sub">news + price overlap</div>
        <div class="kpi-trend" style="color:#718096;">training period</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 3 TABS ───────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "ML & Results"])


# ═══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════
with tab1:

    st.markdown('<div class="sec-hdr"> Stock Price & Sentiment Trend</div>', unsafe_allow_html=True)
    col_p, col_s = st.columns(2)

    with col_p:
        if not t_stock_f.empty:
            fig_price = px.area(t_stock_f, x="date", y="close",
                                title=f"{sel} — Closing Price (USD) · {date_range[0]}–{date_range[1]}",
                                labels={"date":"","close":"Price (USD)"},
                                color_discrete_sequence=["#4361ee"])
            fig_price.update_traces(fill='tozeroy',
                                    fillcolor='rgba(67,97,238,0.09)',
                                    line=dict(color='#4361ee', width=2))
            fig_price.update_layout(height=300, **CT,
                                    xaxis=dict(showgrid=False),
                                    yaxis=dict(showgrid=True, gridcolor="#f0f4f8"))
            st.plotly_chart(fig_price, use_container_width=True)

    with col_s:
        if not t_merged_f.empty:
            disp = t_merged_f.copy()
            if len(disp)>60: disp[scol] = disp[scol].rolling(5).mean()
            colors = ["#06d6a0" if v>0.05 else "#ef476f" if v<-0.05 else "#ffd166"
                      for v in disp[scol].fillna(0)]
            fig_sent = go.Figure()
            fig_sent.add_trace(go.Bar(x=disp["date"], y=disp[scol],
                                      marker_color=colors, opacity=0.8))
            fig_sent.update_layout(
                title=f"{sel} — {model} Sentiment Score · 5-Day Avg · {date_range[0]}–{date_range[1]}",
                height=300, **CT,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#f0f4f8", title="Sentiment Score")
            )
            st.plotly_chart(fig_sent, use_container_width=True)

    st.markdown('<div class="sec-hdr"> Sentiment Distribution & Coverage</div>', unsafe_allow_html=True)
    col_pie, col_vol, col_top5 = st.columns(3)

    with col_pie:
        if not t_sent.empty and "sentiment" in t_sent.columns:
            sc = t_sent["sentiment"].value_counts().reset_index()
            sc.columns = ["Sentiment","Count"]
            fig_pie = px.pie(sc, values="Count", names="Sentiment",
                             title=f"{sel} — Headline Sentiment Breakdown",
                             color="Sentiment",
                             color_discrete_map={"Positive":"#06d6a0",
                                                 "Negative":"#ef476f",
                                                 "Neutral":"#ffd166"},
                             hole=0.42)
            fig_pie.update_layout(height=280, **CT,
                                  legend=dict(font=dict(size=10)))
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_vol:
        if not t_sent.empty:
            t_sent_dated = t_sent.dropna(subset=["date"])
            t_sent_dated["year"] = t_sent_dated["date"].dt.year
            vol = t_sent_dated.groupby("year").size().reset_index(name="count")
            fig_vol = px.bar(vol, x="year", y="count",
                             title=f"{sel} — News Volume by Year",
                             labels={"year":"Year","count":"Articles"},
                             color_discrete_sequence=["#4361ee"])
            fig_vol.update_layout(height=280, **CT,
                                  xaxis=dict(showgrid=False, dtick=2),
                                  yaxis=dict(showgrid=True, gridcolor="#f0f4f8"))
            st.plotly_chart(fig_vol, use_container_width=True)

    with col_top5:
        top5 = sens_df.nlargest(5,"sensitivity")
        rows = "".join([
            f'<div class="dr"><span class="dl">{i+1}. {r["ticker"]} — {COMPANY_NAMES.get(r["ticker"],r["ticker"])}</span>'
            f'<span class="dv" style="color:#4361ee;">r={round(r["sensitivity"],3)}</span></div>'
            for i,(_,r) in enumerate(top5.iterrows())
        ])
        sel_rank_df = sens_df.sort_values("sensitivity",ascending=False).reset_index(drop=True)
        sel_rank_n  = sel_rank_df[sel_rank_df["ticker"]==sel].index[0]+1 if sel in sel_rank_df["ticker"].values else "N/A"
        st.markdown(f"""
        <div class="card" style="height:100%;">
            <div class="card-title">Top 5 Most News-Sensitive Stocks</div>
            {rows}
            <div style="margin-top:0.6rem;background:#eff6ff;border-radius:7px;
                        padding:0.4rem 0.6rem;font-size:0.72rem;color:#1e40af;">
                {sel} ranks #{sel_rank_n} overall — r={corr_val}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr"> Event Study — Market Reaction to News</div>', unsafe_allow_html=True)
    col_ev, col_evn = st.columns([2.5, 1])

    with col_ev:
        fig_ev = go.Figure()
        pos_ev = event_df[event_df["event_type"]=="Positive"].sort_values("lag_day")
        neg_ev = event_df[event_df["event_type"]=="Negative"].sort_values("lag_day")
        fig_ev.add_trace(go.Scatter(x=pos_ev["lag_day"], y=pos_ev["avg_return"],
                                    name="Overall Positive", mode="lines+markers",
                                    line=dict(color="#06d6a0",width=2.5),
                                    marker=dict(size=8)))
        fig_ev.add_trace(go.Scatter(x=neg_ev["lag_day"], y=neg_ev["avg_return"],
                                    name="Overall Negative", mode="lines+markers",
                                    line=dict(color="#ef476f",width=2.5),
                                    marker=dict(size=8)))

        if not t_merged.empty and "daily_return" in t_merged.columns:
            t_merged["sent_pos"] = t_merged[scol] > 0.05
            t_merged["sent_neg"] = t_merged[scol] < -0.05
            try:
                stock_ev_data = []
                for lag in range(6):
                    t_sh = t_merged.copy()
                    t_sh["return_lag"] = t_sh["daily_return"].shift(-lag)
                    pos_ret = t_sh[t_sh["sent_pos"]]["return_lag"].mean()
                    neg_ret = t_sh[t_sh["sent_neg"]]["return_lag"].mean()
                    stock_ev_data.append({"lag":lag,"pos":pos_ret,"neg":neg_ret})
                sev = pd.DataFrame(stock_ev_data)
                fig_ev.add_trace(go.Scatter(x=sev["lag"], y=sev["pos"],
                                            name=f"{sel} Positive",
                                            mode="lines+markers",
                                            line=dict(color="#4361ee",width=1.5,dash="dash"),
                                            marker=dict(size=6)))
                fig_ev.add_trace(go.Scatter(x=sev["lag"], y=sev["neg"],
                                            name=f"{sel} Negative",
                                            mode="lines+markers",
                                            line=dict(color="#ffd166",width=1.5,dash="dash"),
                                            marker=dict(size=6)))
            except:
                pass

        fig_ev.add_hline(y=0, line_dash="dot", line_color="#cbd5e0", line_width=1.5)
        fig_ev.update_layout(
            title=f"Avg Return After News Event — Overall (47 Tickers) vs {sel}",
            height=340, **CT,
            xaxis=dict(tickvals=[0,1,2,3,4,5],
                       title="Days After News Event",
                       showgrid=True, gridcolor="#f0f4f8"),
            yaxis=dict(tickformat=".2%", showgrid=True, gridcolor="#f0f4f8"),
            legend=dict(font=dict(size=10), orientation="h", y=1.12)
        )
        st.plotly_chart(fig_ev, use_container_width=True)

    with col_evn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.success(f"**Positive News — Day 0**\n\n**+{round(pos0*100,3)}%** avg return")
        st.error(f"**Negative News — Day 0**\n\n**{round(neg0*100,3)}%** avg return")
        st.info(f"**By Day 2:** effect absorbed\n\nMarket efficiency confirmed")

    st.markdown('<div class="sec-hdr"> Key Findings</div>', unsafe_allow_html=True)
    f1,f2,f3,f4 = st.columns(4)
    findings = [
        ("01", f"Positive news generated <b>+{round(pos0*100,3)}%</b> avg return on Day 0. Negative news caused <b>{round(neg0*100,3)}%</b>. Effect absorbed by Day 2 — Efficient Market Hypothesis confirmed."),
        ("02", f"<b>ASML</b> shows strongest correlation (r=0.418, p&lt;0.001). <b>{sig_n}/47</b> tickers statistically significant. Semiconductor equipment stocks are most news-sensitive."),
        ("03", f"Logistic Regression <b>{round(lr['accuracy']*100,1)}%</b> accuracy, ROC-AUC <b>{round(lr['roc_auc'],3)}</b>. Trained 2009–2019. Tested on unseen 2020 data. <b>+{round((lr['accuracy']-0.5)*100,1)}pp</b> above random baseline."),
        ("04", f"Sentiment correctly predicted price direction <b>{round(100-overall_anom,2)}%</b> of the time across 47 NASDAQ stocks. VADER 5-day rolling average is the most important ML feature."),
    ]
    for col,(num,text) in zip([f1,f2,f3,f4],findings):
        with col:
            st.markdown(f"""<div class="find">
                <div class="fn">{num}</div>
                <div class="ft">{text}</div>
            </div>""", unsafe_allow_html=True)

    if not t_recent.empty:
        st.markdown('<div class="sec-hdr"> May 2026 Headlines</div>', unsafe_allow_html=True)
        hcols = st.columns(2)
        for i,(_,r) in enumerate(t_recent.sort_values("date",ascending=False).head(6).iterrows()):
            hl   = str(r.get("headline",""))[:92]+"..."
            src  = r.get("source","Unknown")
            sc_v = r.get("polarity_score",0)
            sent = r.get("sentiment","Neutral")
            cc   = "nc nc-p" if sent=="Positive" else "nc nc-n" if sent=="Negative" else "nc nc-z"
            bc   = "bp" if sent=="Positive" else "bn" if sent=="Negative" else "bz"
            with hcols[i%2]:
                st.markdown(f"""<div class="{cc}">
                    <div class="nh">{hl}</div>
                    <div class="nm">{src} &nbsp;·&nbsp;
                        <span class="{bc}">{sent} {round(sc_v,2)}</span>
                    </div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 2 — ANALYSIS
# ═══════════════════════════════════════════════
with tab2:

    st.markdown('<div class="sec-hdr"> Sentiment vs Return Correlation</div>', unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        top20 = corr_df.nlargest(20,"vader_correlation").sort_values("vader_correlation")
        top20["label"] = top20["ticker"]+" — "+top20["ticker"].map(COMPANY_NAMES).fillna(top20["ticker"])
        fig_c = px.bar(top20, x="vader_correlation", y="label", orientation="h",
                       title="Top 20 — VADER Correlation with Stock Returns",
                       labels={"vader_correlation":"Pearson r (VADER)","label":""},
                       color="vader_correlation",
                       color_continuous_scale=["#ffd166","#06d6a0"],
                       text_auto=".3f")
        fig_c.update_layout(height=500, **CT, coloraxis_showscale=False,
                            xaxis=dict(showgrid=True, gridcolor="#f0f4f8"))
        st.plotly_chart(fig_c, use_container_width=True)

    with col_c2:
        compare_df = corr_df.nlargest(15,"vader_correlation")
        compare_df["label"] = compare_df["ticker"]+" — "+compare_df["ticker"].map(COMPANY_NAMES).fillna(compare_df["ticker"])
        compare_df = compare_df.sort_values("vader_correlation", ascending=True)
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="TextBlob", y=compare_df["label"],
            x=compare_df["textblob_correlation"],
            orientation="h", marker_color="#7209b7", opacity=0.8
        ))
        fig_cmp.add_trace(go.Bar(
            name="VADER", y=compare_df["label"],
            x=compare_df["vader_correlation"],
            orientation="h", marker_color="#06d6a0", opacity=0.8
        ))
        fig_cmp.update_layout(
            title="TextBlob vs VADER — Sentiment Model Comparison",
            barmode="group", height=500, **CT,
            xaxis=dict(showgrid=True, gridcolor="#f0f4f8", title="Pearson r"),
            legend=dict(font=dict(size=11), orientation="h", y=1.08)
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown('<div class="sec-hdr"> Anomaly Detection</div>', unsafe_allow_html=True)
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        as_ = anomaly_df.groupby("ticker").agg(
            w=("is_anomaly","sum"), t=("is_anomaly","count")
        ).reset_index()
        as_["ap"] = round(as_["w"]/as_["t"]*100,1)
        as_["cp"] = 100 - as_["ap"]
        as_ = as_.sort_values("ap")
        fig_a = go.Figure()
        fig_a.add_trace(go.Bar(name="Correct", x=as_["ticker"], y=as_["cp"],
                               marker_color="#06d6a0",
                               text=[f"{v}%" for v in as_["cp"]],
                               textposition="inside", textfont=dict(size=7)))
        fig_a.add_trace(go.Bar(name="Anomaly", x=as_["ticker"], y=as_["ap"],
                               marker_color="#ef476f",
                               text=[f"{v}%" for v in as_["ap"]],
                               textposition="inside", textfont=dict(size=7)))
        fig_a.update_layout(
            title=f"Prediction Accuracy vs Anomaly Rate — All 47 Tickers (Overall: {round(100-overall_anom,2)}% correct)",
            barmode="stack", height=380, **CT,
            xaxis=dict(showgrid=False, tickangle=45, tickfont=dict(size=8)),
            yaxis=dict(showgrid=True, gridcolor="#f0f4f8", title="Percentage (%)"),
            legend=dict(font=dict(size=11))
        )
        st.plotly_chart(fig_a, use_container_width=True)

    with col_a2:
        if not t_anomaly.empty and "is_anomaly" in t_anomaly.columns:
            t_anomaly["date"] = pd.to_datetime(t_anomaly["date"])
            fig_as = px.scatter(
                t_anomaly.dropna(subset=["avg_vader_score","daily_return"]),
                x="avg_vader_score", y="daily_return",
                color="is_anomaly",
                color_discrete_map={True:"#ef476f", False:"#06d6a0"},
                title=f"{sel} ({company}) — Correct Predictions vs Anomalies",
                labels={"avg_vader_score":"VADER Sentiment Score",
                        "daily_return":"Daily Stock Return",
                        "is_anomaly":"Anomaly"}
            )
            fig_as.add_hline(y=0, line_dash="dot", line_color="#cbd5e0")
            fig_as.add_vline(x=0, line_dash="dot", line_color="#cbd5e0")
            fig_as.update_layout(height=380, **CT)
            st.plotly_chart(fig_as, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 3 — ML & RESULTS
# ═══════════════════════════════════════════════
with tab3:

    st.markdown('<div class="sec-hdr"> ML Model Performance</div>', unsafe_allow_html=True)
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        fig_ml = go.Figure()
        for met,color in zip(["accuracy","precision","recall","roc_auc"],
                              ["#4361ee","#06d6a0","#ffd166","#7209b7"]):
            fig_ml.add_trace(go.Bar(
                name=met.upper().replace("_","-"),
                x=ml_df["model"], y=ml_df[met],
                marker_color=color,
                text=[f"{v:.1%}" for v in ml_df[met]],
                textposition="outside", textfont=dict(size=10)
            ))
        fig_ml.add_hline(y=0.5, line_dash="dot", line_color="#ef476f",
                         annotation_text="Random baseline 50%",
                         annotation_font_size=9)
        fig_ml.update_layout(
            title="Model Performance — Train: 2009-2019 | Test: 2020",
            barmode="group", height=370, **CT,
            xaxis=dict(showgrid=False),
            yaxis=dict(range=[0,1.1], tickformat=".0%",
                       showgrid=True, gridcolor="#f0f4f8"),
            legend=dict(font=dict(size=10))
        )
        st.plotly_chart(fig_ml, use_container_width=True)

    with col_m2:
        feats = ["avg_textblob_score","avg_vader_score","daily_article_count",
                 "textblob_lag_1","vader_lag_1","textblob_roll_5","vader_roll_5",
                 "positive_count","negative_count"]
        avail = [f for f in feats if f in merged_df.columns]
        df_m  = merged_df[avail+["price_direction","date"]].dropna()
        if len(df_m) >= 20:
            train = df_m[df_m["date"]<"2020-01-01"]
            rfm   = RandomForestClassifier(n_estimators=100, random_state=42)
            rfm.fit(train[avail], train["price_direction"])
            imp = pd.DataFrame({"f":avail,"imp":rfm.feature_importances_}).sort_values("imp")
            lbls = {
                "avg_textblob_score":"TextBlob Score",
                "avg_vader_score":"VADER Score",
                "daily_article_count":"News Volume",
                "textblob_lag_1":"TextBlob Yesterday",
                "vader_lag_1":"VADER Yesterday",
                "textblob_roll_5":"TextBlob 5-Day Avg",
                "vader_roll_5":"VADER 5-Day Avg",
                "positive_count":"Positive Articles",
                "negative_count":"Negative Articles"
            }
            imp["label"] = imp["f"].map(lbls)
            fig_imp = px.bar(imp, x="imp", y="label", orientation="h",
                             title="Feature Importance — What Drives ML Prediction?",
                             labels={"imp":"Importance","label":""},
                             color="imp",
                             color_continuous_scale=["#ffd166","#4361ee"],
                             text_auto=".3f")
            fig_imp.update_layout(height=370, **CT, coloraxis_showscale=False,
                                  xaxis=dict(showgrid=True, gridcolor="#f0f4f8"))
            st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('<div class="sec-hdr"> May 2026 Predictions — Real-World Validation</div>', unsafe_allow_html=True)

    pc1, pc2 = st.columns([1, 1.8])

    with pc1:
        st.markdown(f'<div class="card"><div class="card-title">{sel} — {company} · Predictions</div>',
                    unsafe_allow_html=True)
        if not t_pred.empty:
            for _,r in t_pred.sort_values("date",ascending=False).head(8).iterrows():
                cls = "pred-up" if r["signal"]=="UP" else "pred-dn"
                tc  = "#065f46" if r["signal"]=="UP" else "#991b1b"
                arr = "↑" if r["signal"]=="UP" else "↓"
                st.markdown(f"""<div class="{cls}">
                    <span style="font-size:0.75rem;font-weight:600;color:#1a202c;">{r['date']}</span>
                    <div>
                        <span style="font-size:0.8rem;font-weight:800;color:{tc};">{arr} {r['signal']}</span>
                        <span style="font-size:0.68rem;color:#a0aec0;margin-left:5px;">{round(r['confidence']*100,1)}%</span>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info(f"No 2026 predictions for {sel}.")
        st.markdown("</div>", unsafe_allow_html=True)

    with pc2:
        if not pred_df.empty:
            ps = pred_df.groupby("ticker").agg(
                up=("signal", lambda x:(x=="UP").sum()),
                total=("signal","count")
            ).reset_index()
            ps["up_pct"] = round(ps["up"]/ps["total"]*100,1)
            ps["label"]  = ps["ticker"]+" — "+ps["ticker"].map(COMPANY_NAMES).fillna(ps["ticker"])
            ps = ps.sort_values("up_pct", ascending=True)
            fig_p = px.bar(ps, x="up_pct", y="label", orientation="h",
                           title="May 2026 — % of Bullish (UP) Predictions by Ticker",
                           labels={"up_pct":"% Bullish Predictions","label":""},
                           color="up_pct",
                           color_continuous_scale=["#ef476f","#ffd166","#06d6a0"],
                           color_continuous_midpoint=50,
                           text_auto=".1f")
            fig_p.add_vline(x=50, line_dash="dot", line_color="#a0aec0",
                            annotation_text="50% neutral", annotation_font_size=9)
            fig_p.update_layout(height=380, **CT, coloraxis_showscale=False,
                                xaxis=dict(showgrid=True, gridcolor="#f0f4f8",
                                           range=[0,115]))
            st.plotly_chart(fig_p, use_container_width=True)

# ── RAW DATA ─────────────────────────────────────────────
if show_raw:
    st.markdown("---")
    t1,t2,t3 = st.tabs(["Merged Dataset","Correlation","ML Results"])
    with t1:
        st.dataframe(t_merged, use_container_width=True)
        st.download_button("Download CSV",
                           t_merged.to_csv(index=False).encode(),
                           f"{sel}_data.csv","text/csv")
    with t2:
        st.dataframe(corr_df.sort_values("vader_correlation",ascending=False),
                     use_container_width=True)
    with t3:
        st.dataframe(ml_df, use_container_width=True)

# ── FOOTER ───────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <b style="color:white;font-size:0.88rem;">AI News Impact Analyzer</b><br>
    DAMO 699 Capstone Project &nbsp;|&nbsp; University of Niagara Falls Canada &nbsp;|&nbsp;
    Group 3 &nbsp;|&nbsp; June 2026<br>
    Rohit Singh Sidhu · Janvi Trivedi · Bidita Saha · Devansh Patel &nbsp;|&nbsp;
    Supervisor: Dr. William Pourmajidi<br>
    <span style="color:#4a5568;font-size:0.7rem;">
        NLP: TextBlob · VADER &nbsp;|&nbsp; ML: Logistic Regression · Random Forest
    </span>
</div>
""", unsafe_allow_html=True)