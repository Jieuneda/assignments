"""
Timeseries - 시계열 예측 대시보드
모델: SMA, 지수평활(ES), Holt's, Holt-Winter's, ARIMA, AutoARIMA
평가지표: MAE, MSE, RMSE, MAPE
"""

import io
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Timeseries",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; color: #F1F5F9; }
.stApp { background: #0F172A; color: #F1F5F9; }
p, span, div, label { color: #E2E8F0; }

[data-testid="stSidebar"] { background: #1E293B !important; border-right: 1px solid #334155; }
[data-testid="stSidebar"] * { color: #F1F5F9 !important; }
[data-testid="stSidebar"] label { color: #F1F5F9 !important; font-weight: 500; }

h1, h2, h3, h4 { color: #F8FAFC !important; }

[data-testid="metric-container"] {
    background: #1E293B; border: 1px solid #334155;
    border-radius: 12px; padding: 16px;
    box-shadow: 0 4px 24px rgba(59,130,246,0.08);
}
[data-testid="metric-container"] label { color: #CBD5E1 !important; font-size: 0.75rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #3B82F6 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 1.6rem !important; }

.stTabs [data-baseweb="tab-list"] { background: #1E293B; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #CBD5E1 !important; border-radius: 8px; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #3B82F6 !important; color: white !important; }

.stButton > button {
    background: linear-gradient(135deg, #3B82F6, #1D4ED8); color: white;
    border: none; border-radius: 8px; padding: 10px 24px; font-weight: 600;
    transition: all 0.2s; box-shadow: 0 4px 12px rgba(59,130,246,0.3);
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(59,130,246,0.45); }

.stSelectbox label, .stSlider label, .stCheckbox label { color: #F1F5F9 !important; font-weight: 500; }
.stSelectbox > div > div { color: #F1F5F9 !important; background: #0F172A !important; border-color: #334155 !important; }
/* ── selectbox 드롭다운 팝업 전체 강제 오버라이드 ── */
[data-baseweb="popover"] { background: #1E293B !important; }
[data-baseweb="popover"] * { background: #1E293B !important; color: #F1F5F9 !important; }
[data-baseweb="menu"] { background: #1E293B !important; border: 1px solid #334155 !important; }
[data-baseweb="menu"] * { background: #1E293B !important; color: #F1F5F9 !important; }
[data-baseweb="menu"] ul { background: #1E293B !important; }
[data-baseweb="menu"] li { background: #1E293B !important; color: #F1F5F9 !important; }
[data-baseweb="menu"] li:hover { background: #3B82F6 !important; color: #FFFFFF !important; }
[data-baseweb="option"] { background: #1E293B !important; color: #F1F5F9 !important; }
[data-baseweb="option"]:hover { background: #3B82F6 !important; color: #FFFFFF !important; }
[data-baseweb="select"] * { color: #F1F5F9 !important; }
ul[role="listbox"] { background: #1E293B !important; }
ul[role="listbox"] * { background: #1E293B !important; color: #F1F5F9 !important; }
li[role="option"] { background: #1E293B !important; color: #F1F5F9 !important; }
li[role="option"]:hover { background: #3B82F6 !important; color: #FFFFFF !important; }
li[aria-selected="true"] { background: #1D4ED8 !important; color: #FFFFFF !important; }

[data-testid="stFileUploader"] { border: 2px dashed #334155; border-radius: 12px; padding: 10px; }
[data-testid="stFileUploader"] * { color: #E2E8F0 !important; }

details summary { color: #E2E8F0 !important; font-weight: 500; }
[data-testid="stExpander"] { border: 1px solid #334155 !important; border-radius: 10px; }
[data-testid="stDataFrame"] * { color: #E2E8F0 !important; }

.info-card {
    background: linear-gradient(135deg, #1E293B, #162032);
    border: 1px solid #334155; border-left: 3px solid #3B82F6;
    border-radius: 10px; padding: 16px 20px; margin: 10px 0;
    font-size: 0.875rem; color: #E2E8F0;
}
.warn-card {
    background: #1E293B; border: 1px solid #F59E0B; border-left: 3px solid #F59E0B;
    border-radius: 10px; padding: 14px 18px; margin: 8px 0;
    color: #FDE68A; font-size: 0.85rem;
}
.success-card {
    background: #1E293B; border: 1px solid #10B981; border-left: 3px solid #10B981;
    border-radius: 10px; padding: 14px 18px; margin: 8px 0;
    color: #A7F3D0; font-size: 0.85rem;
}
.sec { color: #CBD5E1; font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 6px; font-weight: 700; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0F172A; }
::-webkit-scrollbar-thumb { background: #475569; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# 공통 Plotly 레이아웃
# ─────────────────────────────────────────────────────
def plo(title="", height=420):
    return dict(
        title=dict(text=title, font=dict(color="#F1F5F9", size=15), x=0.02),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0F172A",
        font=dict(color="#94A3B8", family="Space Grotesk"),
        height=height, margin=dict(l=50, r=30, t=50, b=40),
        xaxis=dict(gridcolor="#1E293B", zeroline=False, tickfont=dict(color="#94A3B8")),
        yaxis=dict(gridcolor="#1E293B", zeroline=False, tickfont=dict(color="#94A3B8")),
        legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="#334155",
                    borderwidth=1, font=dict(color="#E2E8F0", size=11)),
        hovermode="x unified",
    )


# ─────────────────────────────────────────────────────
# 전처리
# ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def preprocess(file_bytes, date_col, value_col, freq):
    """
    CSV 전처리:
    - 날짜 파싱, 타입 변환
    - 결측치 선형 보간
    - IQR×2.5 기반 이상치 탐지
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
    df = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds").reset_index(drop=True)

    df = df.set_index("ds").asfreq(freq)
    miss = int(df["y"].isna().sum())
    df["y"] = df["y"].interpolate(method="time").ffill().bfill()
    df = df.reset_index()

    Q1, Q3 = df["y"].quantile(0.25), df["y"].quantile(0.75)
    IQR = Q3 - Q1
    df["is_anomaly"]    = (df["y"] < Q1 - 2.5 * IQR) | (df["y"] > Q3 + 2.5 * IQR)
    df["anomaly_score"] = (df["y"] - df["y"].median()).abs() / (IQR + 1e-9)

    meta = dict(
        n_rows=len(df), missing=miss,
        n_anom=int(df["is_anomaly"].sum()),
        start=str(df["ds"].iloc[0].date()),
        end=str(df["ds"].iloc[-1].date()),
    )
    return df, meta


# ─────────────────────────────────────────────────────
# 학습/검증 분리 (시계열: 순서 유지)
# ─────────────────────────────────────────────────────
def split(df, val_ratio=0.2):
    n = int(len(df) * (1 - val_ratio))
    return df.iloc[:n].copy(), df.iloc[n:].copy()


# ─────────────────────────────────────────────────────
# 평가지표 계산: MAE, MSE, RMSE, MAPE
# ─────────────────────────────────────────────────────
def calc_metrics(actual, pred):
    """
    평가지표:
    - MAE  : 평균절대오차
    - MSE  : 평균제곱오차
    - RMSE : 평균제곱근오차
    - MAPE : 평균절대백분율오차 (실제값=0 제외)
    """
    a, p = np.array(actual, float), np.array(pred, float)
    n = len(a)
    mae  = float(np.mean(np.abs(a - p)))
    mse  = float(np.mean((a - p) ** 2))
    rmse = float(np.sqrt(mse))
    mask = a != 0
    mape = float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.any() else float("nan")
    return {"MAE": round(mae,4), "MSE": round(mse,4), "RMSE": round(rmse,4), "MAPE(%)": round(mape,2)}


# ─────────────────────────────────────────────────────
# 모델 함수들
# ─────────────────────────────────────────────────────

def run_sma(train, n_periods, window=5):
    """
    단순이동평균 (Simple Moving Average)
    예측치 = 최근 n기간 평균
    window: 이동평균 구간
    """
    last_vals = train["y"].iloc[-window:].values
    forecast  = np.full(n_periods, np.mean(last_vals))
    return forecast


def run_es_simple(train, n_periods, alpha=0.3):
    """
    단순지수평활 (Simple Exponential Smoothing)
    ŷ_{t+1} = α·y_t + (1-α)·ŷ_t
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        model = ExponentialSmoothing(train["y"], trend=None, seasonal=None)
        fit   = model.fit(smoothing_level=alpha, optimized=False)
        return fit.forecast(n_periods).values
    except Exception:
        return None


def run_holts(train, n_periods, alpha=0.3, beta=0.05):
    """
    Holt's 선형추세 지수평활 (Holt's Model)
    추세를 반영한 이중 지수평활
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        model = ExponentialSmoothing(train["y"], trend="add", seasonal=None, damped_trend=False)
        fit   = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
        return fit.forecast(n_periods).values
    except Exception:
        return None


def run_holt_winters(train, n_periods, freq, alpha=0.3, beta=0.05, gamma=0.05):
    """
    Holt-Winter's 계절반영 지수평활
    추세 + 계절성 반영
    - Additive: 계절 변화 일정
    - Multiplicative: 계절 변화가 추세에 비례
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    sp = {"D":7,"W":52,"M":12,"MS":12,"Q":4,"QS":4,"H":24}.get(freq, None)
    if sp is None or len(train) <= sp * 2:
        return None
    try:
        model = ExponentialSmoothing(
            train["y"], trend="add", seasonal="add",
            seasonal_periods=sp, damped_trend=True
        )
        fit = model.fit(
            smoothing_level=alpha, smoothing_trend=beta,
            smoothing_seasonal=gamma, optimized=False
        )
        return fit.forecast(n_periods).values
    except Exception:
        try:
            model = ExponentialSmoothing(
                train["y"], trend="add", seasonal="add",
                seasonal_periods=sp, damped_trend=True
            )
            fit = model.fit(optimized=True)
            return fit.forecast(n_periods).values
        except Exception:
            return None


def run_arima(train, n_periods, pdq=(1,1,1)):
    """
    ARIMA(p,d,q)
    비정상성 시계열을 d차분 후 AR+MA 적용
    - p: 자기회귀 차수
    - d: 차분 차수
    - q: 이동평균 차수
    """
    try:
        from sktime.forecasting.arima import ARIMA
        fh    = list(range(1, n_periods + 1))
        model = ARIMA(order=pdq)
        model.fit(pd.Series(train["y"].values))
        pred = model.predict(fh=fh)
        return np.array(pred)
    except Exception:
        try:
            from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
            model = SM_ARIMA(train["y"].values, order=pdq)
            fit   = model.fit()
            return fit.forecast(n_periods)
        except Exception:
            return None


def run_autoarima(train, n_periods, freq):
    """
    AutoARIMA
    AIC/BIC 기준으로 최적 ARIMA 차수 자동 선택
    계절성 있으면 SARIMA까지 탐색
    """
    sp = {"D":7,"W":52,"M":12,"MS":12,"Q":4,"QS":4,"H":24}.get(freq, None)
    try:
        from sktime.forecasting.arima import AutoARIMA
        fh = list(range(1, n_periods + 1))
        if sp and len(train) > sp * 2:
            model = AutoARIMA(max_p=3, max_q=3, sp=sp, suppress_warnings=True)
        else:
            model = AutoARIMA(max_p=3, max_q=3, seasonal=False, suppress_warnings=True)
        model.fit(pd.Series(train["y"].values))
        pred = model.predict(fh=fh)
        return np.array(pred)
    except Exception:
        try:
            import pmdarima as pm
            model = pm.auto_arima(train["y"], seasonal=False, stepwise=True,
                                   error_action="ignore", suppress_warnings=True)
            return model.predict(n_periods)
        except Exception:
            return None


# ─────────────────────────────────────────────────────
# 차트 함수
# ─────────────────────────────────────────────────────

def chart_overview(df):
    """시계열 개요 + 이상치 시각화"""
    fig = go.Figure()
    nm  = df[~df["is_anomaly"]]
    fig.add_trace(go.Scatter(
        x=nm["ds"], y=nm["y"], mode="lines", name="정상 데이터",
        line=dict(color="#3B82F6", width=1.8),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>값: %{y:,.2f}<extra></extra>",
    ))
    an = df[df["is_anomaly"]]
    if len(an):
        fig.add_trace(go.Scatter(
            x=an["ds"], y=an["y"], mode="markers", name="⚠ 이상치",
            marker=dict(color="#F59E0B", size=10, symbol="diamond",
                        line=dict(color="#FCD34D", width=1.5)),
            hovertemplate="🚨 %{x|%Y-%m-%d}<br>값: %{y:,.2f}<extra></extra>",
        ))
    Q1, Q3 = df["y"].quantile(0.25), df["y"].quantile(0.75)
    IQR    = Q3 - Q1
    fig.add_hline(y=Q3+2.5*IQR, line=dict(color="#EF4444",width=1,dash="dot"),
                  annotation_text="상한(IQR×2.5)", annotation_font_color="#EF4444")
    fig.add_hline(y=Q1-2.5*IQR, line=dict(color="#EF4444",width=1,dash="dot"),
                  annotation_text="하한(IQR×2.5)", annotation_font_color="#EF4444")
    fig.update_layout(**plo("📈 시계열 개요 & 이상치 탐지", 420))
    return fig


def chart_forecast(train, val, results, fdates, best):
    """모든 모델 예측 결과 통합 차트"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train["ds"], y=train["y"], mode="lines", name="학습 데이터",
        line=dict(color="#475569", width=1.5),
    ))
    if val is not None and len(val):
        fig.add_trace(go.Scatter(
            x=val["ds"], y=val["y"], mode="lines", name="실제값 (검증)",
            line=dict(color="#94A3B8", width=2, dash="dot"),
        ))
    COLS = {
        "SMA":             "#64748B",
        "단순ES":          "#A78BFA",
        "Holt's":          "#34D399",
        "Holt-Winter's":   "#FB923C",
        "ARIMA":           "#F472B6",
        "AutoARIMA":       "#FBBF24",
    }
    for nm, data in results.items():
        if data is None: continue
        ib  = (nm == best)
        fc  = data["forecast"]
        dt  = list(fdates[:len(fc)])
        fig.add_trace(go.Scatter(
            x=dt, y=fc,
            mode="lines+markers" if ib else "lines",
            name=f"{'🏆 ' if ib else ''}{nm}",
            line=dict(
                color="#3B82F6" if ib else COLS.get(nm,"#888"),
                width=3 if ib else 1.5,
                dash="solid" if ib else "dot",
            ),
            marker=dict(size=5) if ib else {},
        ))
    sp = val["ds"].iloc[0] if (val is not None and len(val)) else fdates[0]
    fig.add_vline(x=str(sp), line=dict(color="#334155",width=1.5,dash="dash"))
    fig.update_layout(**plo("🔮 모델별 예측 결과 비교", 480))
    return fig


def chart_compare(mdf):
    """RMSE / MAPE 성능 비교 바 차트"""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["RMSE (낮을수록 좋음)", "MAPE % (낮을수록 좋음)"],
                        horizontal_spacing=0.12)
    COLS = {
        "SMA":"#64748B","단순ES":"#A78BFA","Holt's":"#34D399",
        "Holt-Winter's":"#FB923C","ARIMA":"#F472B6","AutoARIMA":"#FBBF24",
    }
    v  = mdf.dropna(subset=["RMSE","MAPE(%)"])
    bc = [COLS.get(m,"#3B82F6") for m in v["Model"]]
    fig.add_trace(go.Bar(
        x=v["Model"], y=v["RMSE"], marker_color=bc,
        text=[f"{x:.4f}" for x in v["RMSE"]], textposition="outside",
        textfont=dict(color="#E2E8F0",size=11),
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=v["Model"], y=v["MAPE(%)"], marker_color=bc,
        text=[f"{x:.2f}%" for x in v["MAPE(%)"]],textposition="outside",
        textfont=dict(color="#E2E8F0",size=11),
    ), row=1, col=2)
    lay = plo("⚡ 모델 성능 비교", 400)
    lay["showlegend"] = False
    fig.update_layout(**lay)
    for ann in fig.layout.annotations:
        ann.font.color="#CBD5E1"; ann.font.size=13
    for ax in [fig.layout.xaxis,fig.layout.xaxis2,fig.layout.yaxis,fig.layout.yaxis2]:
        ax.gridcolor="#1E293B"; ax.tickfont=dict(color="#94A3B8")
    return fig


def chart_decomp(df, freq):
    """시계열 분해: 추세 + 계절성 + 잔차"""
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        sp = {"D":7,"W":52,"M":12,"MS":12,"Q":4,"QS":4,"H":24}.get(freq,7)
        if len(df) < sp * 2 + 1:
            return None
        res = seasonal_decompose(df.set_index("ds")["y"], model="additive", period=sp)
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["원시 데이터", "추세 (Trend)", "계절성 (Seasonal)", "잔차 (Residual)"],
            vertical_spacing=0.12,
        )
        for i, (comp, col) in enumerate([
            (res.observed,"#3B82F6"),(res.trend,"#10B981"),
            (res.seasonal,"#A78BFA"),(res.resid,"#F59E0B"),
        ], 1):
            fig.add_trace(go.Scatter(
                x=comp.index, y=comp.values, mode="lines",
                line=dict(color=col,width=1.5), showlegend=False,
            ), row=i, col=1)
        lay = plo("🔬 시계열 분해 (가법 모형, Additive)", 900)
        fig.update_layout(**lay)
        for i in range(1, 5):
            fig.update_xaxes(gridcolor="#1E293B",tickfont=dict(color="#94A3B8"),row=i,col=1)
            fig.update_yaxes(gridcolor="#1E293B",tickfont=dict(color="#94A3B8"),row=i,col=1)
        for ann in fig.layout.annotations:
            ann.font.color="#CBD5E1"
        return fig
    except Exception:
        return None


def chart_whatif(orig, scen, fdates, train, label):
    """What-If 시나리오 비교 차트"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train["ds"].tail(30), y=train["y"].tail(30),
        mode="lines", name="과거 (최근 30)",
        line=dict(color="#475569",width=1.5),
    ))
    dt = list(fdates[:len(orig)])
    fig.add_trace(go.Scatter(
        x=dt, y=orig, mode="lines+markers", name="기본 예측",
        line=dict(color="#3B82F6",width=2.5), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=dt, y=scen, mode="lines+markers", name=f"🔄 {label}",
        line=dict(color="#F59E0B",width=2.5,dash="dot"),
        marker=dict(size=5, symbol="diamond"),
    ))
    n = min(len(orig), len(scen))
    fig.add_trace(go.Scatter(
        x=dt[:n]+dt[:n][::-1],
        y=list(np.maximum(orig[:n],scen[:n]))+list(np.minimum(orig[:n],scen[:n])[::-1]),
        fill="toself", fillcolor="rgba(245,158,11,0.12)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    fig.update_layout(**plo("🔄 What-If 시나리오 시뮬레이션", 420))
    return fig


# ─────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center;padding:20px 0 10px 0;">
    <div style="font-size:2.2rem;">📈</div>
    <div style="font-size:1.3rem;font-weight:700;color:#F1F5F9;">Timeseries</div>
    <div style="font-size:0.7rem;color:#64748B;letter-spacing:0.12em;text-transform:uppercase;">
        산업데이터공학과 C421086 김지은
    </div>
</div>
<hr style="border-color:#334155;margin:12px 0 20px 0;">
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sec">📂 데이터 업로드</div>', unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader("CSV 파일 선택", type=["csv"], label_visibility="collapsed")

# 변수 기본값
date_col = value_col = freq = None
horizon  = 30
val_ratio = 0.2
run_btn  = False
raw_bytes = None
all_cols  = []

# SMA window
sma_window = 5

# ES 파라미터 기본값
es_alpha = 0.3
holts_alpha = 0.3; holts_beta = 0.05
hw_alpha = 0.3;   hw_beta = 0.05; hw_gamma = 0.05

# ARIMA 차수
arima_p = 1; arima_d = 1; arima_q = 1

if uploaded is not None:
    raw_bytes = uploaded.read()
    try:
        _prev    = pd.read_csv(io.BytesIO(raw_bytes))
        all_cols = _prev.columns.tolist()
    except Exception as e:
        st.sidebar.error(f"CSV 읽기 실패: {e}")
        all_cols = []
    if len(all_cols) < 2:
        st.sidebar.error("열이 최소 2개(날짜+값) 필요합니다.")
        all_cols = []

if all_cols:
    # ── 열 설정 ──
    st.sidebar.markdown('<div class="sec" style="margin-top:16px;">📋 열 설정</div>', unsafe_allow_html=True)
    _kw = ["date","time","ds","날짜","일자","datetime","timestamp"]
    _di = next((i for i,c in enumerate(all_cols) if any(k in c.lower() for k in _kw)), 0)
    date_col  = st.sidebar.selectbox("날짜 열", all_cols, index=_di, key="sel_date")
    val_opts  = [c for c in all_cols if c != date_col]
    value_col = st.sidebar.selectbox("값 열",   val_opts, index=0, key="sel_val")

    st.sidebar.markdown('<div class="sec" style="margin-top:16px;">⏱ 데이터 빈도</div>', unsafe_allow_html=True)
    FREQ_MAP = {"일별":"D","주별":"W","월별":"MS","분기별":"QS","시간별":"H"}
    freq = FREQ_MAP[st.sidebar.selectbox("빈도", list(FREQ_MAP.keys()), index=0, key="sel_freq")]

    st.sidebar.markdown('<div class="sec" style="margin-top:16px;">🎯 예측 설정</div>', unsafe_allow_html=True)
    horizon   = st.sidebar.slider("예측 기간", 7, 180, 30, key="sl_hor")
    val_ratio = st.sidebar.slider("검증 비율 (%)", 10, 40, 20, key="sl_val") / 100

    # ── 모델 선택 ──
    st.sidebar.markdown('<div class="sec" style="margin-top:16px;">🤖 모델 선택</div>', unsafe_allow_html=True)
    use_sma  = st.sidebar.checkbox("SMA (단순이동평균)",         value=True, key="ck_sma")
    use_es   = st.sidebar.checkbox("단순 지수평활 (ES)",          value=True, key="ck_es")
    use_holt = st.sidebar.checkbox("Holt's (선형추세 ES)",        value=True, key="ck_holt")
    use_hw   = st.sidebar.checkbox("Holt-Winter's (계절반영 ES)", value=True, key="ck_hw")
    use_arima= st.sidebar.checkbox("ARIMA",                      value=True, key="ck_arima")
    use_auto = st.sidebar.checkbox("AutoARIMA (자동 차수 선택)",   value=True, key="ck_auto")

    # ── 모델 파라미터 ──
    with st.sidebar.expander("⚙️ 모델 파라미터 설정"):
        st.markdown("**SMA**")
        sma_window = st.slider("이동평균 구간 (n)", 2, 24, 5, key="sma_w")

        st.markdown("**단순 ES**")
        es_alpha = st.slider("α (평활계수)", 0.01, 1.0, 0.3, step=0.05, key="es_a")

        st.markdown("**Holt's**")
        holts_alpha = st.slider("α (level)", 0.01, 1.0, 0.3, step=0.05, key="h_a")
        holts_beta  = st.slider("β (trend)", 0.01, 1.0, 0.05, step=0.05, key="h_b")

        st.markdown("**Holt-Winter's**")
        hw_alpha = st.slider("α (level)",    0.01, 1.0, 0.3,  step=0.05, key="hw_a")
        hw_beta  = st.slider("β (trend)",    0.01, 1.0, 0.05, step=0.05, key="hw_b")
        hw_gamma = st.slider("γ (seasonal)", 0.01, 1.0, 0.05, step=0.05, key="hw_g")

        st.markdown("**ARIMA(p,d,q)**")
        arima_p = st.slider("p (AR 차수)", 0, 5, 1, key="ar_p")
        arima_d = st.slider("d (차분 차수)", 0, 2, 1, key="ar_d")
        arima_q = st.slider("q (MA 차수)", 0, 5, 1, key="ar_q")

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.sidebar.button("🚀 분석 실행", use_container_width=True, key="run")

st.sidebar.markdown("""
<hr style="border-color:#334155;margin:20px 0 14px 0;">
<div style="font-size:0.72rem;color:#475569;text-align:center;line-height:1.9;">
    SMA · ES · Holt's · Holt-Winter's · ARIMA<br>
    <span style="color:#3B82F6;">MAE · MSE · RMSE · MAPE</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# 메인 화면
# ─────────────────────────────────────────────────────
st.markdown("""
<div style="padding:10px 0 24px 0;">
<h1 style="font-size:2rem;font-weight:700;margin:0;letter-spacing:-0.02em;">
    📈 Timeseries
    <span style="font-size:0.95rem;color:#64748B;font-weight:400;margin-left:12px;">
        산업데이터공학과 C421086 김지은
    </span>
</h1>
</div>
""", unsafe_allow_html=True)

# 랜딩 화면
if uploaded is None:
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="info-card">
        <b style="color:#3B82F6;">📘 평활법 모델</b><br><br>
        SMA, 단순 지수평활(ES), Holt's 모델,
        Holt-Winter's 모델을 적용하여 예측합니다.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="info-card">
        <b style="color:#A78BFA;">📘 ARIMA 모델</b><br><br>
        ARIMA(p,d,q)와 AutoARIMA로 정상성 검정 후
        최적 차수를 자동 선택하여 예측합니다.
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="info-card">
        <b style="color:#10B981;">📘 평가지표</b><br><br>
        MAE, MSE, RMSE, MAPE 기준으로
        모델 성능을 비교하고 최적 모델을 선택합니다.
        </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:60px 0;">
        <div style="font-size:3rem;margin-bottom:16px;">📂</div>
        <div style="font-size:1.1rem;color:#64748B;">좌측 사이드바에서 CSV 파일을 업로드하세요</div>
        <div style="font-size:0.85rem;color:#334155;margin-top:8px;">날짜 컬럼 + 값 컬럼이 있는 단변량 시계열 CSV</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

if date_col is None or value_col is None or raw_bytes is None:
    st.warning("CSV 열을 인식하지 못했습니다. 파일을 다시 업로드하세요.")
    st.stop()

# 전처리
with st.spinner("⚙️ 전처리 중..."):
    try:
        df, meta = preprocess(raw_bytes, date_col, value_col, freq)
    except Exception as e:
        st.error(f"전처리 실패: {e}")
        st.stop()

# 요약 메트릭
st.markdown("### 📊 데이터 요약")
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("총 포인트",   f"{meta['n_rows']:,}")
c2.metric("보간 결측치", f"{meta['missing']:,}")
c3.metric("이상치 수",   f"{meta['n_anom']:,}")
c4.metric("평균",        f"{df['y'].mean():,.2f}")
c5.metric("표준편차",    f"{df['y'].std():,.2f}")
c6.metric("기간",        f"{meta['start'][:7]}~{meta['end'][:7]}")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────────────
t1,t2,t3,t4,t5 = st.tabs([
    "📈 개요 & 이상치", "🔮 예측 결과", "⚡ 모델 비교", "🔬 시계열 분해", "🔄 What-If",
])

# ── TAB 1: 개요 & 이상치 ─────────────────────────────
with t1:
    st.plotly_chart(chart_overview(df), use_container_width=True)

    if meta["n_anom"]:
        st.markdown(f'<div class="warn-card">⚠️ <b>{meta["n_anom"]}개 이상치</b> 감지 — IQR × 2.5 기준</div>',
                    unsafe_allow_html=True)
        with st.expander("🔍 이상치 상세"):
            adf = (df[df["is_anomaly"]][["ds","y","anomaly_score"]]
                   .rename(columns={"ds":"날짜","y":"값","anomaly_score":"이상치 점수"})
                   .sort_values("이상치 점수", ascending=False))
            st.dataframe(adf.head(20).style.format({"값":"{:,.4f}","이상치 점수":"{:.2f}"}),
                         use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="success-card">✅ 이상치 없음 — 데이터 품질 양호</div>', unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        fh = go.Figure(go.Histogram(x=df["y"], nbinsx=40, marker_color="#3B82F6", opacity=0.8))
        fh.update_layout(**plo("📊 값 분포 히스토그램", 300))
        st.plotly_chart(fh, use_container_width=True)
    with cb:
        fb = go.Figure(go.Box(y=df["y"], name="값", marker_color="#3B82F6",
                               line_color="#60A5FA", fillcolor="rgba(59,130,246,0.2)"))
        fb.update_layout(**plo("📦 박스플롯", 300))
        st.plotly_chart(fb, use_container_width=True)


# ── TAB 2: 예측 결과 ─────────────────────────────────
with t2:
    if not run_btn:
        st.markdown("""<div class="info-card" style="text-align:center;padding:40px;">
        🚀 사이드바의 <b style="color:#3B82F6;">분석 실행</b> 버튼을 누르세요
        </div>""", unsafe_allow_html=True)
    else:
        tr, vl   = split(df, val_ratio)
        fdates   = pd.date_range(start=df["ds"].iloc[-1], periods=horizon+1, freq=freq)[1:]
        vl_len   = len(vl)
        results  = {}
        mlist    = []
        bar = st.progress(0, text="모델 학습 중...")
        step = 0
        total_models = sum([use_sma, use_es, use_holt, use_hw, use_arima, use_auto])
        step_size = 90 // max(total_models, 1)

        # SMA
        if use_sma:
            bar.progress(step:=step+step_size, text="SMA 학습 중...")
            fc = run_sma(tr, vl_len+horizon, sma_window)
            if fc is not None:
                mk = calc_metrics(vl["y"].values, fc[:vl_len])
                mlist.append({"Model":"SMA", **mk})
                results["SMA"] = {"forecast": fc[vl_len:vl_len+horizon], "metrics": mk}

        # 단순 ES
        if use_es:
            bar.progress(step:=step+step_size, text="단순 지수평활 학습 중...")
            fc = run_es_simple(tr, vl_len+horizon, es_alpha)
            if fc is not None:
                mk = calc_metrics(vl["y"].values, fc[:vl_len])
                mlist.append({"Model":"단순ES", **mk})
                results["단순ES"] = {"forecast": fc[vl_len:vl_len+horizon], "metrics": mk}

        # Holt's
        if use_holt:
            bar.progress(step:=step+step_size, text="Holt's 모델 학습 중...")
            fc = run_holts(tr, vl_len+horizon, holts_alpha, holts_beta)
            if fc is not None:
                mk = calc_metrics(vl["y"].values, fc[:vl_len])
                mlist.append({"Model":"Holt's", **mk})
                results["Holt's"] = {"forecast": fc[vl_len:vl_len+horizon], "metrics": mk}

        # Holt-Winter's
        if use_hw:
            bar.progress(step:=step+step_size, text="Holt-Winter's 학습 중...")
            fc = run_holt_winters(tr, vl_len+horizon, freq, hw_alpha, hw_beta, hw_gamma)
            if fc is not None:
                mk = calc_metrics(vl["y"].values, fc[:vl_len])
                mlist.append({"Model":"Holt-Winter's", **mk})
                results["Holt-Winter's"] = {"forecast": fc[vl_len:vl_len+horizon], "metrics": mk}

        # ARIMA
        if use_arima:
            bar.progress(step:=step+step_size, text=f"ARIMA({arima_p},{arima_d},{arima_q}) 학습 중...")
            fc = run_arima(tr, vl_len+horizon, (arima_p, arima_d, arima_q))
            if fc is not None:
                mk = calc_metrics(vl["y"].values, fc[:vl_len])
                mlist.append({"Model":"ARIMA", **mk})
                results["ARIMA"] = {"forecast": fc[vl_len:vl_len+horizon], "metrics": mk}

        # AutoARIMA
        if use_auto:
            bar.progress(step:=step+step_size, text="AutoARIMA 학습 중...")
            fc = run_autoarima(tr, vl_len+horizon, freq)
            if fc is not None:
                mk = calc_metrics(vl["y"].values, fc[:vl_len])
                mlist.append({"Model":"AutoARIMA", **mk})
                results["AutoARIMA"] = {"forecast": fc[vl_len:vl_len+horizon], "metrics": mk}

        bar.progress(100, text="✅ 완료!"); bar.empty()

        mdf  = pd.DataFrame(mlist)
        best = mdf.loc[mdf["RMSE"].idxmin(),"Model"] if len(mdf) else list(results.keys())[0] if results else "없음"
        st.session_state.update(dict(mdf=mdf, results=results, fdates=fdates, tr=tr, vl=vl,
                                     best=best, horizon=horizon))

        br = mdf[mdf["Model"]==best].iloc[0]
        st.markdown(f"""<div class="success-card">
        🏆 <b>최적 모델: {best}</b> —
        RMSE: {br['RMSE']:.4f} &nbsp;|&nbsp;
        MAE: {br['MAE']:.4f} &nbsp;|&nbsp;
        MAPE: {br['MAPE(%)']:.2f}%
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(chart_forecast(tr, vl, results, fdates, best), use_container_width=True)

        if best in results:
            fcv = results[best]["forecast"]
            with st.expander("📋 예측값 상세 테이블"):
                st.dataframe(
                    pd.DataFrame({"날짜": fdates[:len(fcv)], f"예측값({best})": fcv.round(4)}),
                    use_container_width=True, hide_index=True,
                )


# ── TAB 3: 모델 비교 ─────────────────────────────────
with t3:
    if "mdf" not in st.session_state:
        st.markdown('<div class="info-card" style="text-align:center;padding:40px;">분석을 먼저 실행하세요</div>',
                    unsafe_allow_html=True)
    else:
        mdf  = st.session_state["mdf"]
        best = st.session_state["best"]

        st.plotly_chart(chart_compare(mdf), use_container_width=True)

        # 성능 순위 테이블
        st.markdown("#### 📊 전체 성능 지표")
        d2 = mdf.copy()
        d2["순위"] = d2["RMSE"].rank().astype(int)
        d2["🏆"]  = d2["Model"].apply(lambda x: "🏆" if x==best else "")
        st.dataframe(
            d2[["순위","🏆","Model","MAE","MSE","RMSE","MAPE(%)"]].sort_values("순위")
            .style.format({"MAE":"{:.4f}","MSE":"{:.4f}","RMSE":"{:.4f}","MAPE(%)":"{:.2f}%"})
            .background_gradient(subset=["RMSE"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True,
        )

        st.markdown("""<div class="info-card">
        <b>📘 평가지표 해설</b><br><br>
        • <b>MAE</b> (평균절대오차): 오차 절대값의 평균. 이상치에 덜 민감. MAE=0이면 완벽한 예측<br>
        • <b>MSE</b> (평균제곱오차): 오차 제곱의 평균. 통계학의 분산과 유사. 큰 오차에 더 민감<br>
        • <b>RMSE</b> (평균제곱근오차): MSE의 제곱근. 예측치와 단위 동일. 실무에서 가장 많이 사용<br>
        • <b>MAPE</b> (평균절대백분율오차): 실제값 대비 오차 비율(%). 서로 다른 척도 간 비교 가능
        </div>""", unsafe_allow_html=True)


# ── TAB 4: 시계열 분해 ───────────────────────────────
with t4:
    st.markdown("""<div class="info-card">
    <b>📘 시계열 분해법</b> — 추세(Trend) + 계절성(Seasonal) + 잔차(Residual)로 분리<br>
    가법 모형: Y = 추세 + 계절성 + 잔차 &nbsp;|&nbsp;
    Holt-Winter's 예측은 이 세 성분을 합산하여 계산
    </div>""", unsafe_allow_html=True)

    fd = chart_decomp(df, freq)
    if fd:
        st.plotly_chart(fd, use_container_width=True)
    else:
        sp = {"D":7,"W":52,"M":12,"MS":12,"Q":4,"QS":4,"H":24}.get(freq,7)
        st.markdown(f"""<div class="warn-card">
        ⚠️ 분해를 위해 최소 <b>{sp*2+1}개</b> 포인트 필요 (현재 {len(df)}개)<br>
        더 긴 기간의 데이터를 사용하거나 빈도를 변경하세요.
        </div>""", unsafe_allow_html=True)


# ── TAB 5: What-If 시나리오 ──────────────────────────
with t5:
    if "results" not in st.session_state:
        st.markdown('<div class="info-card" style="text-align:center;padding:40px;">분석을 먼저 실행하세요</div>',
                    unsafe_allow_html=True)
    else:
        best    = st.session_state["best"]
        results = st.session_state["results"]
        fdates  = st.session_state["fdates"]
        tr      = st.session_state["tr"]
        hor     = st.session_state["horizon"]

        st.markdown('<div class="info-card">최적 모델 예측에 외부 충격을 가정해 예측치 변화를 시뮬레이션합니다.</div>',
                    unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)
        with s1: stype = st.selectbox("충격 유형", ["비율 변화 (%)", "절대값 변화", "선형 트렌드"])
        with s2: sval  = st.number_input("충격 크기", value=10.0, step=1.0)
        with s3: sst   = st.slider("시작 시점 (N번째 예측부터)", 1, max(1,hor-1), 1)

        orig = results[best]["forecast"].copy().astype(float)
        scen = orig.copy()
        sl   = slice(sst-1, None)

        if stype == "비율 변화 (%)":
            scen[sl] = orig[sl] * (1 + sval/100)
            label = f"{sval:+.1f}% 충격"
        elif stype == "절대값 변화":
            scen[sl] = orig[sl] + sval
            label = f"절대값 {sval:+.1f}"
        else:
            t    = np.arange(len(orig))
            scen = orig + np.where(t >= sst-1, sval*(t-(sst-1)), 0)
            label = f"선형 트렌드 +{sval:.1f}/기간"

        st.plotly_chart(chart_whatif(orig, scen, fdates, tr, label), use_container_width=True)

        diff = scen - orig
        w1, w2, w3 = st.columns(3)
        w1.metric("평균 변화량",  f"{diff.mean():+.4f}",
                  delta=f"{diff.mean()/(orig.mean()+1e-9)*100:+.2f}%")
        w2.metric("누적 합 변화", f"{diff.sum():+.2f}")
        w3.metric("최대 절대차",  f"{np.abs(diff).max():.4f}")

        with st.expander("📋 시나리오 비교 테이블"):
            st.dataframe(pd.DataFrame({
                "날짜":       fdates[:len(orig)],
                "기본 예측":  orig.round(4),
                "시나리오":   scen.round(4),
                "차이":       diff.round(4),
                "변화율(%)":  ((scen/(orig+1e-9)-1)*100).round(2),
            }), use_container_width=True, hide_index=True)