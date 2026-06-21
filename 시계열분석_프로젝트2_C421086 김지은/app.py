"""
Time Series Anomaly Detection Dashboard
수업 내용 기반: Darts ForecastingAnomalyModel + KMeansScorer + NormScorer
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import time

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TimeSeries AD",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1421 50%, #0a1628 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111827 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #63b3ed;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(90deg, rgba(99,179,237,0.08) 0%, rgba(159,122,234,0.08) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(99,179,237,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #63b3ed, #9f7aea, #68d391);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}
.app-header p {
    color: #94a3b8;
    font-size: 0.9rem;
    margin: 0;
    font-weight: 400;
}
.badge {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 99px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-right: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}
.metric-card {
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 12px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(99,179,237,0.35); }
.metric-card .label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .value.danger { color: #fc8181; }
.metric-card .value.warning { color: #f6ad55; }
.metric-card .value.success { color: #68d391; }
.metric-card .accent-bar {
    position: absolute;
    top: 0; left: 0;
    width: 3px;
    height: 100%;
    border-radius: 12px 0 0 12px;
}

/* ── Section Title ── */
.section-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #63b3ed;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 24px 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,179,237,0.3), transparent);
}

/* ── Alert Box ── */
.alert-box {
    background: rgba(252,129,129,0.08);
    border: 1px solid rgba(252,129,129,0.25);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 12px 0;
    color: #fca5a5;
    font-size: 0.875rem;
}
.info-box {
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 12px 0;
    color: #93c5fd;
    font-size: 0.875rem;
}
.success-box {
    background: rgba(104,211,145,0.08);
    border: 1px solid rgba(104,211,145,0.25);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 12px 0;
    color: #86efac;
    font-size: 0.875rem;
}

/* ── Plotly charts dark ── */
.js-plotly-plot .plotly .modebar {
    background: transparent !important;
}

/* ── Stacked info ── */
.detail-row {
    display: flex;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid rgba(99,179,237,0.07);
    font-size: 0.85rem;
}
.detail-row .dk { color: #64748b; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
.detail-row .dv { color: #e2e8f0; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }

/* ── Streamlit overrides ── */
.stButton button {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.88 !important; }

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

.stProgress > div > div { background: linear-gradient(90deg, #2563eb, #7c3aed) !important; }

[data-testid="stFileUploader"] {
    border: 2px dashed rgba(99,179,237,0.25) !important;
    border-radius: 12px !important;
    background: rgba(15,23,42,0.4) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,23,42,0.6) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid rgba(99,179,237,0.12) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    color: #64748b !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    padding: 6px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,179,237,0.15) !important;
    color: #63b3ed !important;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly Dark Theme Helper ────────────────────────────────────────────────────
CHART_BG = "rgba(10,14,26,0)"
GRID_COLOR = "rgba(99,179,237,0.08)"
ACCENT_COLORS = ["#63b3ed", "#9f7aea", "#68d391", "#f6ad55", "#fc8181", "#76e4f7"]

def dark_layout(fig, title="", height=380):
    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor="rgba(13,19,33,0.7)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        title=dict(text=title, font=dict(color="#e2e8f0", size=13, family="Inter"), x=0.01, pad=dict(t=8)),
        height=height,
        margin=dict(l=50, r=20, t=45, b=40),
        legend=dict(
            bgcolor="rgba(13,19,33,0.8)",
            bordercolor="rgba(99,179,237,0.2)",
            borderwidth=1,
            font=dict(size=10),
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR, gridwidth=1,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR, gridwidth=1,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(size=10),
        ),
    )
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 TimeSeries AD")
    st.markdown("<div style='height:1px;background:rgba(99,179,237,0.15);margin:8px 0 16px'></div>", unsafe_allow_html=True)

    st.markdown("**📁 데이터 업로드**")
    uploaded_file = st.file_uploader(
        "CSV 파일을 드래그하거나 선택",
        type=["csv"],
        help="시간 컬럼 + 수치형 피처 컬럼이 있는 다변량 CSV",
    )

    st.markdown("<div style='height:1px;background:rgba(99,179,237,0.15);margin:16px 0'></div>", unsafe_allow_html=True)
    st.markdown("**⚙️ 모델 설정**")

    train_ratio = st.slider("학습 데이터 비율", 0.5, 0.85, 0.7, 0.05,
                            help="전체 데이터 중 학습에 사용할 비율")

    scorer_options = st.multiselect(
        "Anomaly Scorer 선택",
        ["NormScorer (예측오차 기반)", "KMeansScorer k=2 (클러스터링)", "KMeansScorer k=3"],
        default=["NormScorer (예측오차 기반)", "KMeansScorer k=2 (클러스터링)"],
        help="복수의 Scorer를 결합하여 다각도 탐지 수행"
    )

    lag_weeks = st.slider("Lag (과거 참조 기간, 포인트 수)", 12, 336, 48,
                          help="예측 모델이 참조할 과거 시점 수")

    threshold_pct = st.slider("이상 임계값 백분위 (%)", 80, 99, 95,
                              help="이 백분위 이상의 점수를 이상으로 판정")

    st.markdown("<div style='height:1px;background:rgba(99,179,237,0.15);margin:16px 0'></div>", unsafe_allow_html=True)
    st.markdown("**📖 수업 이론 기반**")
    st.markdown("""
<div style='font-size:0.75rem;color:#64748b;line-height:1.7'>
🔵 <b style='color:#63b3ed'>ForecastingAnomalyModel</b><br>
예측 모델 + Scorer 결합<br><br>
🟣 <b style='color:#9f7aea'>NormScorer</b><br>
예측 오차 절댓값 기반<br><br>
🟢 <b style='color:#68d391'>KMeansScorer</b><br>
클러스터링 기반 이상 점수<br><br>
🟡 <b style='color:#f6ad55'>ThresholdDetector</b><br>
점수 → 이진 이상 변환
</div>
""", unsafe_allow_html=True)

# ── Main Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <span class="badge">Darts</span>
    <span class="badge">SKLearnModel</span>
    <span class="badge">ForecastingAnomalyModel</span>
    <h1>시계열 이상 탐지 대시보드</h1>
    <p>다변량 시계열 CSV를 업로드하면 자동으로 이상 탐지를 수행합니다 — NormScorer + KMeansScorer 기반 앙상블 탐지</p>
</div>
""", unsafe_allow_html=True)

# ── Load & Parse ───────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df

def detect_time_column(df):
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col])
            if parsed.is_monotonic_increasing:
                return col
        except:
            continue
    return None

def infer_freq(dt_index):
    if len(dt_index) < 3:
        return None
    diffs = dt_index[1:] - dt_index[:-1]
    mode_diff = pd.Series(diffs).mode()[0]
    return pd.tseries.frequencies.to_offset(mode_diff)

# ── If no file: show demo guide ────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
<div class="info-box">
    📂 <b>시작하기:</b> 왼쪽 사이드바에서 CSV 파일을 업로드하세요.<br>
    CSV 형식: <code style="font-family:JetBrains Mono">timestamp, feature1, feature2, ...</code> — 시간 컬럼과 하나 이상의 수치형 피처가 필요합니다.
</div>
""", unsafe_allow_html=True)

    # Sample data preview
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown('<div class="section-title">📋 예시 CSV 형식</div>', unsafe_allow_html=True)
        sample_preview = pd.DataFrame({
            "timestamp": ["2022-01-01 00:00", "2022-01-01 01:00", "2022-01-01 02:00", "..."],
            "cpu_usage": [50.99, 52.50, 56.67, "..."],
            "memory_usage": [65.72, 67.40, 66.80, "..."],
            "network_traffic": [191.37, 212.63, 225.18, "..."],
        })
        st.dataframe(sample_preview, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-title">🔬 탐지 파이프라인</div>', unsafe_allow_html=True)
        pipeline_steps = [
            ("1", "CSV 파싱 & TimeSeries 변환", "#63b3ed"),
            ("2", "Train / Test 분할", "#9f7aea"),
            ("3", "SKLearnModel 학습 (lags + cyclic covariates)", "#68d391"),
            ("4", "ForecastingAnomalyModel 구성", "#f6ad55"),
            ("5", "NormScorer + KMeansScorer 점수 산출", "#fc8181"),
            ("6", "임계값 기반 이상 구간 탐지 및 시각화", "#76e4f7"),
        ]
        for num, step, color in pipeline_steps:
            st.markdown(f"""
<div style='display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid rgba(99,179,237,0.07)'>
    <span style='background:{color}22;border:1px solid {color}44;color:{color};
                 font-family:JetBrains Mono;font-size:0.7rem;font-weight:700;
                 width:22px;height:22px;border-radius:50%;display:flex;align-items:center;
                 justify-content:center;flex-shrink:0'>{num}</span>
    <span style='font-size:0.82rem;color:#94a3b8'>{step}</span>
</div>""", unsafe_allow_html=True)

    # Theory quick cards
    st.markdown('<div class="section-title">📚 수업 핵심 이론</div>', unsafe_allow_html=True)
    theory_cols = st.columns(3)
    theories = [
        ("점이상 (Point Anomaly)", "전역 이상(Global): 전체를 통틀어 정상 범주를 벗어나는 값\n맥락 이상(Contextual): 인접 맥락 고려 시 이상인 값", "#63b3ed"),
        ("패턴이상 (Pattern Anomaly)", "모양 이상(Shapelet): 주기/모양 흐름과 다른 부분 시계열\n계절성 이상(Seasonal): 계절 주기를 벗어난 패턴\n추세 이상(Trend): 추세에 영구적 변화를 주는 구간", "#9f7aea"),
        ("Darts AD 모듈 구조", "Scorers: 이상 점수 계산 (NormScorer, KMeansScorer)\nDetectors: 점수 → 이진 시계열 변환\nAggregators: 다변량 → 단변량 축소\nAnomaly Models: 예측 모델 + Scorer 결합", "#68d391"),
    ]
    for col, (title, desc, color) in zip(theory_cols, theories):
        with col:
            st.markdown(f"""
<div style='background:rgba(15,23,42,0.6);border:1px solid {color}28;border-top:3px solid {color};
            border-radius:10px;padding:16px 18px;height:100%'>
    <div style='font-size:0.78rem;font-weight:700;color:{color};margin-bottom:8px'>{title}</div>
    <div style='font-size:0.78rem;color:#94a3b8;line-height:1.7;white-space:pre-line'>{desc}</div>
</div>""", unsafe_allow_html=True)

    st.stop()

# ── File loaded: process ────────────────────────────────────────────────────────
file_bytes = uploaded_file.read()
file_key = hash(file_bytes)  # detect file changes

df_raw = load_csv(file_bytes)

# Detect time column
time_col = detect_time_column(df_raw)
if time_col is None:
    st.markdown('<div class="alert-box">❌ <b>시간 컬럼을 자동으로 감지하지 못했습니다.</b> 날짜/시간 형식의 컬럼이 포함되어 있는지 확인하세요.</div>', unsafe_allow_html=True)
    st.stop()

numeric_cols = [c for c in df_raw.columns if c != time_col and pd.api.types.is_numeric_dtype(df_raw[c])]
if len(numeric_cols) == 0:
    st.markdown('<div class="alert-box">❌ <b>수치형 피처 컬럼이 없습니다.</b></div>', unsafe_allow_html=True)
    st.stop()

df_raw[time_col] = pd.to_datetime(df_raw[time_col])
df_raw = df_raw.sort_values(time_col).reset_index(drop=True)
df_raw = df_raw.dropna(subset=[time_col])

# ── Sidebar: column selector ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='height:1px;background:rgba(99,179,237,0.15);margin:16px 0'></div>", unsafe_allow_html=True)
    st.markdown("**📊 분석 컬럼 선택**")
    selected_cols = st.multiselect(
        "이상 탐지할 피처",
        numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))],
        help="분석할 수치형 피처를 선택하세요"
    )

if not selected_cols:
    st.markdown('<div class="alert-box">⚠️ 사이드바에서 분석할 피처를 선택하세요.</div>', unsafe_allow_html=True)
    st.stop()

# ── Data Overview Cards ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 데이터 개요</div>', unsafe_allow_html=True)

freq_guess = infer_freq(pd.DatetimeIndex(df_raw[time_col]))
n_rows = len(df_raw)
n_features = len(selected_cols)
date_range_str = f"{df_raw[time_col].iloc[0].strftime('%Y-%m-%d')} ~ {df_raw[time_col].iloc[-1].strftime('%Y-%m-%d')}"

m1, m2, m3, m4 = st.columns(4)
cards = [
    ("TOTAL ROWS", f"{n_rows:,}", "#63b3ed", "success"),
    ("FEATURES", str(n_features), "#9f7aea", "success"),
    ("FREQUENCY", str(freq_guess) if freq_guess else "불규칙", "#68d391", "success"),
    ("DATE RANGE", date_range_str, "#f6ad55", "warning"),
]
for col, (label, val, color, cls) in zip([m1, m2, m3, m4], cards):
    with col:
        st.markdown(f"""
<div class="metric-card">
    <div class="accent-bar" style="background:{color}"></div>
    <div class="label">{label}</div>
    <div class="value {cls}" style="font-size:{'1.1rem' if len(val)>12 else '1.5rem'}">{val}</div>
</div>""", unsafe_allow_html=True)

# ── Run Detection ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🚀 이상 탐지 실행</div>', unsafe_allow_html=True)

run_col, info_col = st.columns([1, 3])
with run_col:
    run_btn = st.button("▶ 탐지 시작", use_container_width=True)
with info_col:
    scorer_info = " + ".join([s.split(" ")[0] for s in scorer_options])
    st.markdown(f"""
<div class="info-box" style="margin:0">
    🔧 <b>설정 요약:</b> 학습비율 {train_ratio*100:.0f}% · Scorer: {scorer_info if scorer_info else '미선택'} · Lag: {lag_weeks} · 임계값: {threshold_pct}th 백분위
</div>""", unsafe_allow_html=True)

# ── Session state for results ───────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "last_file_key" not in st.session_state:
    st.session_state.last_file_key = None

# Reset if file changed
if st.session_state.last_file_key != file_key:
    st.session_state.results = None
    st.session_state.last_file_key = file_key

if run_btn or st.session_state.results is not None:
    if run_btn or st.session_state.results is None:
        # ── Build TimeSeries & run detection ────────────────────────────────────
        with st.spinner(""):
            prog_bar = st.progress(0, text="TimeSeries 객체 변환 중...")
            status_box = st.empty()

            try:
                from darts import TimeSeries
                from darts.ad import ForecastingAnomalyModel, KMeansScorer, NormScorer
                from darts.dataprocessing.transformers import Scaler
                from darts.models import SKLearnModel

                # ── 1. Build TimeSeries ──────────────────────────────────────────
                df_ts = df_raw[[time_col] + selected_cols].copy()
                df_ts = df_ts.set_index(time_col)
                df_ts = df_ts.dropna()

                # Infer freq more robustly
                try:
                    df_ts.index.freq = pd.infer_freq(df_ts.index)
                except:
                    pass
                if df_ts.index.freq is None:
                    freq_str = freq_guess if freq_guess else "h"
                    df_ts = df_ts.resample(str(freq_str)).mean().interpolate()

                ts_full = TimeSeries.from_dataframe(df_ts)
                prog_bar.progress(15, text="Train/Test 분할 중...")

                # ── 2. Train/Test split ──────────────────────────────────────────
                split_idx = int(len(ts_full) * train_ratio)
                if split_idx < lag_weeks + 10:
                    split_idx = lag_weeks + 10
                ts_train = ts_full[:split_idx]
                ts_test = ts_full[split_idx:]

                prog_bar.progress(25, text="Scaler 정규화 중...")

                # ── 3. Scale ─────────────────────────────────────────────────────
                scaler = Scaler()
                ts_train_sc = scaler.fit_transform(ts_train)
                ts_test_sc = scaler.transform(ts_test)
                ts_full_sc = scaler.transform(ts_full)

                prog_bar.progress(35, text="공변량(Cyclic Encoders) 생성 중...")

                # ── 4. Covariates (cyclic: hour, dayofweek) ──────────────────────
                add_encoders = {"cyclic": {"future": ["hour", "dayofweek"]}}

                prog_bar.progress(45, text="SKLearnModel 학습 중 (예측 기반 이상 탐지)...")

                # ── 5. Forecasting model ─────────────────────────────────────────
                from sklearn.ensemble import RandomForestRegressor
                lags_to_use = min(lag_weeks, split_idx - 2)

                forecasting_model = SKLearnModel(
                    lags=lags_to_use,
                    lags_future_covariates=[0],
                    output_chunk_length=1,
                    add_encoders=add_encoders,
                    model=RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
                )
                forecasting_model.fit(ts_train_sc)
                prog_bar.progress(65, text="ForecastingAnomalyModel + Scorers 결합 중...")

                # ── 6. Scorers ───────────────────────────────────────────────────
                scorers = []
                scorer_names = []
                if any("NormScorer" in s for s in scorer_options):
                    scorers.append(NormScorer())
                    scorer_names.append("NormScorer")
                if any("k=2" in s for s in scorer_options):
                    scorers.append(KMeansScorer(k=2, window=12))
                    scorer_names.append("KMeansScorer(k=2)")
                if any("k=3" in s for s in scorer_options):
                    scorers.append(KMeansScorer(k=3, window=12))
                    scorer_names.append("KMeansScorer(k=3)")

                if not scorers:
                    scorers = [NormScorer()]
                    scorer_names = ["NormScorer"]

                anomaly_model = ForecastingAnomalyModel(
                    model=forecasting_model,
                    scorer=scorers,
                )
                prog_bar.progress(75, text="Scorer 학습 중...")
                anomaly_model.fit(ts_train_sc, allow_model_training=False)

                prog_bar.progress(85, text="이상 점수 계산 중...")
                anomaly_scores_list = anomaly_model.score(ts_test_sc)

                # score() returns a tuple when multiple scorers, or single TimeSeries
                if isinstance(anomaly_scores_list, tuple):
                    anomaly_scores_list = list(anomaly_scores_list)
                elif not isinstance(anomaly_scores_list, list):
                    anomaly_scores_list = [anomaly_scores_list]

                prog_bar.progress(95, text="결과 집계 및 시각화 준비 중...")

                # ── 7. Aggregate scores ──────────────────────────────────────────
                score_dfs = []
                for i, (sc_ts, sc_name) in enumerate(zip(anomaly_scores_list, scorer_names)):
                    sc_df = sc_ts.to_dataframe()
                    sc_df.columns = [f"{sc_name}_{c}" for c in sc_df.columns]
                    score_dfs.append(sc_df)

                all_scores_df = pd.concat(score_dfs, axis=1)

                # Combined score (mean-normalize then average)
                norm_scores = (all_scores_df - all_scores_df.min()) / (all_scores_df.max() - all_scores_df.min() + 1e-9)
                combined_score = norm_scores.mean(axis=1)

                # Threshold
                threshold_val = np.percentile(combined_score.values, threshold_pct)
                anomaly_flags = combined_score > threshold_val

                prog_bar.progress(100, text="완료!")
                time.sleep(0.3)
                prog_bar.empty()
                status_box.empty()

                # ── Save results ─────────────────────────────────────────────────
                st.session_state.results = {
                    "ts_full": ts_full,
                    "ts_train": ts_train,
                    "ts_test": ts_test,
                    "ts_full_sc": ts_full_sc,
                    "ts_train_sc": ts_train_sc,
                    "ts_test_sc": ts_test_sc,
                    "all_scores_df": all_scores_df,
                    "combined_score": combined_score,
                    "anomaly_flags": anomaly_flags,
                    "threshold_val": threshold_val,
                    "scorer_names": scorer_names,
                    "split_idx": split_idx,
                    "df_raw": df_raw,
                    "selected_cols": selected_cols,
                    "time_col": time_col,
                }

                st.markdown('<div class="success-box">✅ <b>이상 탐지 완료!</b> 아래 대시보드에서 결과를 확인하세요.</div>', unsafe_allow_html=True)

            except Exception as e:
                prog_bar.empty()
                status_box.empty()
                st.markdown(f'<div class="alert-box">❌ <b>오류 발생:</b> {str(e)}</div>', unsafe_allow_html=True)
                with st.expander("상세 오류 보기"):
                    import traceback
                    st.code(traceback.format_exc())
                st.stop()

# ── Results Dashboard ───────────────────────────────────────────────────────────
if st.session_state.results is None:
    st.stop()

R = st.session_state.results
combined_score = R["combined_score"]
anomaly_flags = R["anomaly_flags"]
threshold_val = R["threshold_val"]
all_scores_df = R["all_scores_df"]
ts_full = R["ts_full"]
ts_train = R["ts_train"]
ts_test = R["ts_test"]
selected_cols = R["selected_cols"]
scorer_names = R["scorer_names"]
split_idx = R["split_idx"]

# Anomaly stats
n_anomalies = int(anomaly_flags.sum())
anomaly_ratio = n_anomalies / len(anomaly_flags) * 100
test_len = len(ts_test)

# ── Result Metric Cards ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📈 탐지 결과 요약</div>', unsafe_allow_html=True)

rc1, rc2, rc3, rc4 = st.columns(4)
res_cards = [
    ("ANOMALIES FOUND", str(n_anomalies), "danger" if n_anomalies > 0 else "success", "#fc8181"),
    ("ANOMALY RATIO", f"{anomaly_ratio:.1f}%", "warning" if anomaly_ratio > 5 else "success", "#f6ad55"),
    ("TEST WINDOW", f"{test_len:,} pts", "success", "#68d391"),
    ("THRESHOLD", f"{threshold_val:.4f}", "success", "#9f7aea"),
]
for col, (label, val, cls, color) in zip([rc1, rc2, rc3, rc4], res_cards):
    with col:
        st.markdown(f"""
<div class="metric-card">
    <div class="accent-bar" style="background:{color}"></div>
    <div class="label">{label}</div>
    <div class="value {cls}">{val}</div>
</div>""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌊 시계열 & 이상 구간",
    "📊 이상 점수 분석",
    "🔬 피처별 분석",
    "📉 점수 분포",
    "🗂 원시 데이터"
])

# ────────────────────────────────────────────────────────────────────────────────
# TAB 1: Time Series + Anomaly Overlay
# ────────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">전체 시계열 + 탐지된 이상 구간</div>', unsafe_allow_html=True)

    # Get anomaly time ranges (contiguous)
    anomaly_times = combined_score.index[anomaly_flags]

    for feat_idx, feat in enumerate(selected_cols):
        df_full = ts_full.to_dataframe()
        if feat not in df_full.columns:
            continue

        fig = go.Figure()

        # Train region background — use add_shape to avoid Timestamp arithmetic bug in add_vrect
        train_start_str = str(ts_full.start_time())
        train_end_str = str(ts_train.end_time())
        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=train_start_str, x1=train_end_str,
            y0=0, y1=1,
            fillcolor="rgba(99,179,237,0.04)", line_width=0, layer="below",
        )
        fig.add_annotation(
            x=train_start_str, xref="x", yref="paper",
            y=0.97, text="학습 구간", showarrow=False,
            font=dict(size=10, color="#63b3ed"), xanchor="left",
        )

        # Anomaly shading
        if len(anomaly_times) > 0:
            anomaly_groups = []
            group_start = anomaly_times[0]
            prev_t = anomaly_times[0]
            step_secs = (anomaly_times[1] - anomaly_times[0]).total_seconds() if len(anomaly_times) > 1 else 3600
            for t in anomaly_times[1:]:
                if (t - prev_t).total_seconds() > 2 * step_secs:
                    anomaly_groups.append((group_start, prev_t))
                    group_start = t
                prev_t = t
            anomaly_groups.append((group_start, prev_t))

            for i, (gs, ge) in enumerate(anomaly_groups):
                fig.add_shape(
                    type="rect", xref="x", yref="paper",
                    x0=str(gs), x1=str(ge), y0=0, y1=1,
                    fillcolor="rgba(252,129,129,0.15)",
                    line=dict(color="rgba(252,129,129,0.4)", width=1),
                    layer="below",
                )
                if i == 0:
                    fig.add_annotation(
                        x=str(gs), xref="x", yref="paper",
                        y=0.97, text="⚠", showarrow=False,
                        font=dict(size=13, color="#fc8181"), xanchor="left",
                    )

        # Train line
        train_df = ts_train.to_dataframe()
        if feat in train_df.columns:
            fig.add_trace(go.Scatter(
                x=train_df.index, y=train_df[feat],
                mode="lines", name="학습 데이터",
                line=dict(color=ACCENT_COLORS[feat_idx % len(ACCENT_COLORS)], width=1.2, dash="solid"),
                opacity=0.6,
            ))

        # Test line
        test_df = ts_test.to_dataframe()
        if feat in test_df.columns:
            fig.add_trace(go.Scatter(
                x=test_df.index, y=test_df[feat],
                mode="lines", name="테스트 데이터",
                line=dict(color=ACCENT_COLORS[feat_idx % len(ACCENT_COLORS)], width=1.8),
            ))

        # Anomaly points
        anom_mask = anomaly_flags.reindex(test_df.index, fill_value=False)
        anom_x = test_df.index[anom_mask]
        anom_y = test_df[feat][anom_mask]
        if len(anom_x) > 0:
            fig.add_trace(go.Scatter(
                x=anom_x, y=anom_y,
                mode="markers", name="이상 포인트",
                marker=dict(color="#fc8181", size=6, symbol="circle",
                           line=dict(color="#fff", width=1)),
            ))

        dark_layout(fig, title=f"피처: {feat}", height=300)
        st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# TAB 2: Anomaly Score Analysis
# ────────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">이상 점수 시계열 (결합 점수 + 개별 Scorer)</div>', unsafe_allow_html=True)

    # Combined score plot
    fig_score = go.Figure()

    # Threshold line
    fig_score.add_hline(
        y=threshold_val, line_dash="dash",
        line_color="#fc8181", line_width=1.5,
        annotation_text=f"임계값 ({threshold_pct}th pct)", annotation_position="bottom right",
        annotation_font=dict(color="#fc8181", size=10),
    )

    # Fill anomaly zones
    fig_score.add_trace(go.Scatter(
        x=combined_score.index,
        y=np.where(combined_score > threshold_val, combined_score, threshold_val),
        mode="lines", fill="tonexty",
        fillcolor="rgba(252,129,129,0.15)",
        line=dict(color="rgba(252,129,129,0)", width=0),
        showlegend=False, hoverinfo="skip",
    ))

    # Combined score line
    fig_score.add_trace(go.Scatter(
        x=combined_score.index, y=combined_score.values,
        mode="lines", name="Combined Score",
        line=dict(color="#f6ad55", width=2),
        fill="tozeroy", fillcolor="rgba(246,173,85,0.05)",
    ))

    dark_layout(fig_score, "결합 이상 점수 (정규화 평균)", height=280)
    st.plotly_chart(fig_score, use_container_width=True)

    # Individual scorer plots
    if len(scorer_names) > 1:
        st.markdown('<div class="section-title">개별 Scorer 점수 비교</div>', unsafe_allow_html=True)
        score_cols = st.columns(min(len(scorer_names), 2))
        for i, sc_name in enumerate(scorer_names):
            sc_cols = [c for c in all_scores_df.columns if c.startswith(sc_name)]
            if not sc_cols:
                continue
            sc_data = all_scores_df[sc_cols].mean(axis=1)
            fig_ind = go.Figure()
            fig_ind.add_trace(go.Scatter(
                x=sc_data.index, y=sc_data.values,
                mode="lines", name=sc_name,
                line=dict(color=ACCENT_COLORS[i], width=1.5),
                fill="tozeroy", fillcolor=["rgba(99,179,237,0.09)","rgba(159,122,234,0.09)","rgba(104,211,145,0.09)","rgba(246,173,85,0.09)","rgba(252,129,129,0.09)","rgba(118,228,247,0.09)"][i % 6],
            ))
            dark_layout(fig_ind, title=sc_name, height=220)
            with score_cols[i % 2]:
                st.plotly_chart(fig_ind, use_container_width=True)

    # Anomaly event table
    st.markdown('<div class="section-title">탐지된 이상 이벤트</div>', unsafe_allow_html=True)
    if n_anomalies > 0:
        anom_df_display = pd.DataFrame({
            "시각": combined_score.index[anomaly_flags],
            "이상 점수": combined_score[anomaly_flags].round(5).values,
            "판정": ["⚠️ 이상" for _ in range(n_anomalies)],
        }).reset_index(drop=True)
        anom_df_display["순위"] = anom_df_display["이상 점수"].rank(ascending=False).astype(int)
        anom_df_display = anom_df_display.sort_values("이상 점수", ascending=False)
        st.dataframe(
            anom_df_display.head(50),
            use_container_width=True,
            hide_index=True,
            column_config={
                "이상 점수": st.column_config.ProgressColumn(
                    "이상 점수", min_value=0, max_value=float(combined_score.max()), format="%.5f"
                )
            }
        )
    else:
        st.markdown('<div class="success-box">✅ 이상 포인트가 탐지되지 않았습니다. 임계값 또는 Scorer 설정을 조정해보세요.</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# TAB 3: Per-Feature Analysis
# ────────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">피처별 통계 및 분포 분석</div>', unsafe_allow_html=True)

    df_full_pd = ts_full.to_dataframe()
    df_test_pd = ts_test.to_dataframe()

    for feat in selected_cols:
        if feat not in df_full_pd.columns:
            continue
        col_a, col_b = st.columns([1.8, 1])
        with col_a:
            # Box plot: train vs test vs anomaly
            fig_box = go.Figure()
            train_vals = ts_train.to_dataframe()[feat].dropna() if feat in ts_train.to_dataframe() else pd.Series()
            test_vals = df_test_pd[feat].dropna() if feat in df_test_pd else pd.Series()
            anom_mask_test = anomaly_flags.reindex(df_test_pd.index, fill_value=False)
            anom_vals = df_test_pd[feat][anom_mask_test].dropna() if feat in df_test_pd else pd.Series()

            for vals, name, color in [
                (train_vals, "학습 구간", "#63b3ed"),
                (test_vals, "테스트 구간", "#9f7aea"),
                (anom_vals, "이상 포인트", "#fc8181"),
            ]:
                if len(vals) > 0:
                    fig_box.add_trace(go.Box(
                        y=vals, name=name,
                        marker_color=color,
                        line_color=color,
                        boxmean="sd",
                    ))
            dark_layout(fig_box, title=f"{feat} — 분포 비교", height=280)
            st.plotly_chart(fig_box, use_container_width=True)

        with col_b:
            st.markdown(f"<div style='margin-top:40px'></div>", unsafe_allow_html=True)
            series_all = df_full_pd[feat].dropna()
            stats = {
                "평균": f"{series_all.mean():.3f}",
                "표준편차": f"{series_all.std():.3f}",
                "최솟값": f"{series_all.min():.3f}",
                "최댓값": f"{series_all.max():.3f}",
                "25th pct": f"{series_all.quantile(0.25):.3f}",
                "75th pct": f"{series_all.quantile(0.75):.3f}",
            }
            for k, v in stats.items():
                st.markdown(f"""
<div class="detail-row">
    <span class="dk">{k}</span>
    <span class="dv">{v}</span>
</div>""", unsafe_allow_html=True)

        # Rolling stats
        fig_roll = go.Figure()
        roll_mean = df_full_pd[feat].rolling(24).mean()
        roll_std = df_full_pd[feat].rolling(24).std()

        fig_roll.add_trace(go.Scatter(
            x=df_full_pd.index, y=roll_mean + 2 * roll_std,
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_roll.add_trace(go.Scatter(
            x=df_full_pd.index, y=roll_mean - 2 * roll_std,
            mode="lines", fill="tonexty",
            fillcolor="rgba(99,179,237,0.08)", line=dict(width=0),
            name="±2σ 밴드", hoverinfo="skip",
        ))
        fig_roll.add_trace(go.Scatter(
            x=df_full_pd.index, y=df_full_pd[feat],
            mode="lines", name="원본",
            line=dict(color="#64748b", width=1), opacity=0.7,
        ))
        fig_roll.add_trace(go.Scatter(
            x=df_full_pd.index, y=roll_mean,
            mode="lines", name="이동평균(24)",
            line=dict(color="#63b3ed", width=1.8),
        ))

        dark_layout(fig_roll, title=f"{feat} — 롤링 평균 & ±2σ 밴드", height=240)
        st.plotly_chart(fig_roll, use_container_width=True)
        st.markdown("<hr style='border:0;border-top:1px solid rgba(99,179,237,0.08);margin:8px 0 20px'>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# TAB 4: Score Distribution
# ────────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">이상 점수 분포 및 평가 지표</div>', unsafe_allow_html=True)

    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        # Histogram of combined score
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=combined_score.values,
            nbinsx=60,
            marker=dict(
                color=combined_score.values,
                colorscale=[[0, "#1a2744"], [0.5, "#63b3ed"], [1, "#fc8181"]],
                line=dict(width=0),
            ),
            name="점수 분포",
        ))
        fig_hist.add_vline(
            x=threshold_val, line_dash="dash",
            line_color="#fc8181", line_width=2,
            annotation_text=f"임계값\n{threshold_val:.4f}",
            annotation_font=dict(color="#fc8181", size=10),
        )
        dark_layout(fig_hist, "결합 이상 점수 히스토그램", height=300)
        st.plotly_chart(fig_hist, use_container_width=True)

    with dist_col2:
        # Percentile curve
        pcts = np.arange(0, 101, 1)
        pct_vals = np.percentile(combined_score.values, pcts)
        fig_pct = go.Figure()
        fig_pct.add_trace(go.Scatter(
            x=pcts, y=pct_vals,
            mode="lines", name="백분위 곡선",
            line=dict(color="#9f7aea", width=2),
            fill="tozeroy", fillcolor="rgba(159,122,234,0.08)",
        ))
        fig_pct.add_trace(go.Scatter(
            x=[threshold_pct], y=[threshold_val],
            mode="markers", name=f"{threshold_pct}th pct",
            marker=dict(color="#fc8181", size=10, symbol="x"),
        ))
        dark_layout(fig_pct, "이상 점수 백분위 분포", height=300)
        fig_pct.update_xaxes(title_text="백분위 (%)")
        fig_pct.update_yaxes(title_text="점수")
        st.plotly_chart(fig_pct, use_container_width=True)

    # Anomaly type classification
    st.markdown('<div class="section-title">이상 유형 분류 (이론 기반 자동 판정)</div>', unsafe_allow_html=True)

    if n_anomalies > 0:
        anom_scores_sorted = combined_score[anomaly_flags].sort_values(ascending=False)
        global_threshold = np.percentile(combined_score.values, 99)
        context_threshold = threshold_val

        n_global = int((anom_scores_sorted > global_threshold).sum())
        n_context = int(n_anomalies - n_global)

        type_col1, type_col2, type_col3 = st.columns(3)

        type_cards = [
            ("전역 이상\n(Global Anomaly)", n_global,
             "전체 시계열을 통틀어 정상 범주를\n크게 벗어나는 데이터 포인트", "#fc8181"),
            ("맥락 이상\n(Contextual Anomaly)", n_context,
             "값 자체보다 인접 시계열의\n맥락을 고려할 때 이상인 포인트", "#f6ad55"),
            ("정상 포인트", len(combined_score) - n_anomalies,
             "설정된 임계값 이하로\n정상 범주 내에 있는 포인트", "#68d391"),
        ]
        for col, (type_name, cnt, desc, color) in zip([type_col1, type_col2, type_col3], type_cards):
            with col:
                st.markdown(f"""
<div style='background:rgba(15,23,42,0.7);border:1px solid {color}28;border-top:3px solid {color};
            border-radius:10px;padding:18px;text-align:center'>
    <div style='font-size:0.75rem;font-weight:700;color:{color};margin-bottom:8px;white-space:pre-line'>{type_name}</div>
    <div style='font-size:2rem;font-weight:800;color:{color};font-family:JetBrains Mono'>{cnt:,}</div>
    <div style='font-size:0.72rem;color:#64748b;margin-top:8px;white-space:pre-line'>{desc}</div>
</div>""", unsafe_allow_html=True)

        # Scorer comparison radar / bar
        st.markdown('<div class="section-title">Scorer별 기여도 비교</div>', unsafe_allow_html=True)
        scorer_means = {}
        for sc_name in scorer_names:
            sc_cols = [c for c in all_scores_df.columns if c.startswith(sc_name)]
            if sc_cols:
                scorer_means[sc_name] = float(all_scores_df[sc_cols].values.mean())

        if scorer_means:
            fig_bar = go.Figure(go.Bar(
                x=list(scorer_means.keys()),
                y=list(scorer_means.values()),
                marker=dict(
                    color=ACCENT_COLORS[:len(scorer_means)],
                    line=dict(width=0),
                ),
            ))
            dark_layout(fig_bar, "Scorer별 평균 이상 점수", height=250)
            st.plotly_chart(fig_bar, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# TAB 5: Raw Data
# ────────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">원시 데이터 미리보기</div>', unsafe_allow_html=True)

    df_display = df_raw[[time_col] + selected_cols].copy().reset_index(drop=True)

    # Build anomaly flag column safely — match timestamps by string value to avoid index mismatch
    anomaly_timestamps = set(combined_score.index[anomaly_flags].astype(str))
    df_display["이상 여부"] = df_display[time_col].astype(str).isin(anomaly_timestamps)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={
            "이상 여부": st.column_config.CheckboxColumn("⚠️ 이상 여부")
        }
    )

    # Download buttons
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        csv_data = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 결과 CSV 다운로드",
            data=csv_data,
            file_name="anomaly_detection_result.csv",
            mime="text/csv",
        )
    with dl_col2:
        score_df_export = combined_score.to_frame(name="anomaly_score")
        score_df_export["is_anomaly"] = anomaly_flags.values
        score_csv = score_df_export.to_csv().encode("utf-8")
        st.download_button(
            label="📥 이상 점수 CSV 다운로드",
            data=score_csv,
            file_name="anomaly_scores.csv",
            mime="text/csv",
        )

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:40px;padding:20px 0;border-top:1px solid rgba(99,179,237,0.1);
     text-align:center;color:#334155;font-size:0.78rem'>
    시계열 이상 탐지 대시보드 &nbsp;·&nbsp; Darts ForecastingAnomalyModel + NormScorer + KMeansScorer &nbsp;·&nbsp;
    <span style='color:#1e3a5f'>수업 이론 기반 구현</span>
</div>
""", unsafe_allow_html=True)