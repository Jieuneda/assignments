import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import norm
import io, warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ProcessIQ · 공정분석",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════
# DESIGN — 라이트 프리미엄 테마
# 배경: 순백 + 연회색 카드 / 포인트: 딥 인디고 + 에메랄드 + 앰버
# 폰트: Syne(헤딩) + DM Sans(본문) + JetBrains Mono(수치)
# 시그니처: 카드 상단 4px 컬러 보더 + 수평 타임라인 레이아웃
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background: #f5f4f0 !important;
  color: #1a1a2e !important;
}

.stApp { background: #f5f4f0 !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: #1a1a2e !important;
  border-right: none !important;
}

/* 모든 사이드바 텍스트 흰색 */
[data-testid="stSidebar"],
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span,
[data-testid="stSidebar"] [class*="css"] {
  color: #e8eeff !important;
  opacity: 1 !important;
}

/* 입력 필드 — 텍스트 입력 영역만 */
[data-testid="stSidebar"] .stNumberInput input[type="number"],
[data-testid="stSidebar"] .stNumberInput > div > div > input {
  background: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.3) !important;
  color: #0f172a !important;
  border-radius: 6px 0 0 6px !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  -webkit-text-fill-color: #0f172a !important;
}

/* 셀렉트박스 */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [role="listbox"] * {
  background: rgba(255,255,255,0.1) !important;
  color: #e8eeff !important;
  border-color: rgba(255,255,255,0.2) !important;
}

/* 라디오 버튼 */
[data-testid="stSidebar"] .stRadio > div { gap: 6px !important; }
[data-testid="stSidebar"] .stRadio label {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  border-radius: 8px !important;
  padding: 8px 14px !important;
  font-size: 0.85rem !important;
  color: #e8eeff !important;
  cursor: pointer;
  transition: all .15s;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(99,102,241,0.3) !important;
  border-color: rgba(99,102,241,0.5) !important;
}

/* 슬라이더 */
[data-testid="stSidebar"] .stSlider > div > div > div { background: rgba(255,255,255,0.15) !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div { background: #6366f1 !important; }

/* 플레이스홀더 */
[data-testid="stSidebar"] input::placeholder { color: rgba(255,255,255,0.4) !important; }

/* 업다운 버튼 */
[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"],
[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"],
[data-testid="stSidebar"] .stNumberInput button {
  color: #e8eeff !important;
  background: rgba(255,255,255,0.08) !important;
  border-color: rgba(255,255,255,0.15) !important;
}

/* ── SIDEBAR BRAND ── */
.sb-brand {
  padding: 24px 0 20px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  margin-bottom: 24px;
}
.sb-brand-name {
  font-family: 'Syne', sans-serif;
  font-size: 1.3rem; font-weight: 800;
  color: #fff; letter-spacing: -0.3px;
}
.sb-brand-sub { font-size: 0.7rem; color: rgba(255,255,255,0.4); margin-top: 2px; letter-spacing: 1px; text-transform: uppercase; }
.sb-section {
  font-size: 0.6rem; font-weight: 600; letter-spacing: 2px;
  text-transform: uppercase; color: rgba(255,255,255,0.3);
  margin: 22px 0 10px; padding-bottom: 6px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.sb-tip {
  background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3);
  border-radius: 8px; padding: 10px 12px; margin-top: 20px;
  font-size: 0.75rem; color: #a5b4fc; line-height: 1.5;
}

/* ── PAGE HEADER ── */
.page-header {
  display: flex; align-items: flex-start; justify-content: space-between;
  margin-bottom: 32px; padding-bottom: 24px;
  border-bottom: 1px solid rgba(26,26,46,0.08);
}
.page-title {
  font-family: 'Syne', sans-serif;
  font-size: 2.6rem; font-weight: 800; color: #1a1a2e;
  letter-spacing: -1.5px; line-height: 1; margin: 0 0 6px;
}
.page-sub { font-size: 0.85rem; color: #64748b; font-weight: 300; }
.page-tag {
  display: inline-block; background: #eef2ff; border: 1px solid #c7d2fe;
  color: #4f46e5; font-size: 0.65rem; font-weight: 600;
  padding: 4px 10px; border-radius: 20px; letter-spacing: 1px;
  text-transform: uppercase; margin-bottom: 8px;
}
.status-pill {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 18px; border-radius: 40px;
  font-size: 0.82rem; font-weight: 600; white-space: nowrap;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; }

/* ── METRIC STRIP ── */
.metric-strip {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;
  margin-bottom: 24px;
}
.metric-card {
  background: #fff; border-radius: 14px;
  border-top: 4px solid transparent;
  padding: 18px 20px 16px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
  transition: transform .15s, box-shadow .15s;
}
.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}
.mc-label { font-size: 0.65rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: #94a3b8; margin-bottom: 8px; }
.mc-val   { font-family: 'JetBrains Mono', monospace; font-size: 1.9rem; font-weight: 500; line-height: 1; margin-bottom: 5px; }
.mc-sub   { font-size: 0.72rem; color: #94a3b8; }
.mc-indigo  { border-top-color: #6366f1; } .mc-indigo  .mc-val { color: #4f46e5; }
.mc-emerald { border-top-color: #10b981; } .mc-emerald .mc-val { color: #059669; }
.mc-amber   { border-top-color: #f59e0b; } .mc-amber   .mc-val { color: #d97706; }
.mc-rose    { border-top-color: #f43f5e; } .mc-rose    .mc-val { color: #e11d48; }
.mc-violet  { border-top-color: #8b5cf6; } .mc-violet  .mc-val { color: #7c3aed; }
.mc-teal    { border-top-color: #14b8a6; } .mc-teal    .mc-val { color: #0d9488; }
.mc-sky     { border-top-color: #0ea5e9; } .mc-sky     .mc-val { color: #0284c7; }
.mc-slate   { border-top-color: #64748b; } .mc-slate   .mc-val { color: #475569; }

/* ── SECTION ── */
.sec {
  background: #fff; border-radius: 16px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
  padding: 24px 26px; margin-bottom: 16px;
}
.sec-title {
  font-family: 'Syne', sans-serif;
  font-size: 0.9rem; font-weight: 700; color: #1a1a2e;
  letter-spacing: -0.2px; margin-bottom: 16px;
  display: flex; align-items: center; gap: 8px;
}
.sec-badge {
  display: inline-block; font-size: 0.62rem; font-weight: 600;
  padding: 2px 8px; border-radius: 10px; letter-spacing: .5px;
  font-family: 'DM Sans', sans-serif; text-transform: uppercase;
}
.sb-blue   { background:#eef2ff; color:#4f46e5; }
.sb-green  { background:#ecfdf5; color:#059669; }
.sb-amber  { background:#fffbeb; color:#d97706; }
.sb-rose   { background:#fff1f2; color:#e11d48; }
.sb-violet { background:#f5f3ff; color:#7c3aed; }
.sb-teal   { background:#f0fdfa; color:#0d9488; }

/* ── FORMULA ── */
.formula-wrap {
  background: #fafafa; border: 1px solid #e2e8f0;
  border-left: 4px solid #6366f1; border-radius: 0 10px 10px 0;
  padding: 14px 18px; font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem; color: #312e81; line-height: 2; margin: 12px 0;
}

/* ── BADGE ── */
.badge { display:inline-block; border-radius:20px; padding:4px 12px; font-size:0.72rem; font-weight:600; }
.b-good { background:#ecfdf5; color:#059669; border:1px solid #a7f3d0; }
.b-warn { background:#fffbeb; color:#d97706; border:1px solid #fde68a; }
.b-bad  { background:#fff1f2; color:#e11d48; border:1px solid #fecdd3; }

/* ── STAT TABLE ── */
.stbl { width:100%; border-collapse:collapse; font-size:0.82rem; }
.stbl td { padding:8px 6px; border-bottom:1px solid #f1f5f9; }
.stbl td:first-child { color:#64748b; }
.stbl td:last-child  { color:#1a1a2e; font-family:'JetBrains Mono',monospace; text-align:right; font-weight:500; }

/* ── VIOLATION ── */
.viol-item {
  background:#fff5f6; border:1px solid #fecdd3;
  border-left:4px solid #f43f5e; border-radius:0 8px 8px 0;
  padding:10px 14px; margin:6px 0;
}
.viol-title { color:#e11d48; font-weight:600; font-size:0.82rem; margin-bottom:2px; }
.viol-body  { color:#64748b; font-size:0.78rem; }

/* ── INFO BOX ── */
.info-b { background:#eef2ff; border:1px solid #c7d2fe; border-radius:8px; padding:10px 14px; font-size:0.8rem; color:#4338ca; margin:8px 0; }
.warn-b { background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:10px 14px; font-size:0.8rem; color:#92400e; margin:8px 0; }
.good-b { background:#ecfdf5; border:1px solid #a7f3d0; border-radius:8px; padding:10px 14px; font-size:0.8rem; color:#065f46; margin:8px 0; }

/* ── REC BOX ── */
.rec-item {
  border-radius:10px; padding:14px 18px; margin:8px 0;
  border-left:4px solid;
}
.rec-title { font-weight:600; margin-bottom:3px; font-size:0.88rem; }
.rec-body  { font-size:0.8rem; line-height:1.5; color:#475569; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background:#f1f5f9 !important; border:1px solid #e2e8f0 !important;
  border-radius:10px !important; padding:3px !important; gap:2px !important;
}
.stTabs [data-baseweb="tab"] {
  background:transparent !important; color:#64748b !important;
  border-radius:8px !important; font-weight:500 !important;
  font-size:0.84rem !important; padding:8px 18px !important;
}
.stTabs [aria-selected="true"] {
  background:#1a1a2e !important; color:#fff !important;
}

/* ── CLEANUP ── */
#MainMenu,footer,header { visibility:hidden; }
.block-container { padding-top:28px !important; padding-bottom:40px !important; }
div[data-testid="stVerticalBlock"] > div { gap:0 !important; }
::-webkit-scrollbar { width:4px; } 
::-webkit-scrollbar-thumb { background:#e2e8f0; border-radius:2px; }

/* ── SIDEBAR ALWAYS VISIBLE + TOGGLE BUTTON ── */
[data-testid="stSidebar"] { display: block !important; visibility: visible !important; }
[data-testid="collapsedControl"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  background: #1a1a2e !important;
  color: white !important;
  border-radius: 0 8px 8px 0 !important;
  width: 32px !important;
  height: 32px !important;
  z-index: 9999 !important;
}

/* ── DIVIDER ── */
.divider { height:1px; background: linear-gradient(90deg,#6366f1,#14b8a6,transparent); margin:20px 0; border:none; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def d2(n):
    T={2:1.128,3:1.693,4:2.059,5:2.326,6:2.534,7:2.704,8:2.847,9:2.970,10:3.078}
    return T.get(n,1.128+(n-2)*0.06)
def c4a(n): return 1-1/(4*(n-1))-7/(32*(n-1)**2) if n>1 else 1.0
def A2(n):
    T={2:1.880,3:1.023,4:0.729,5:0.577,6:0.483,7:0.419,8:0.373,9:0.337,10:0.308}
    return T.get(n,3/(d2(n)*n**0.5))
def A3(n):
    T={2:2.659,3:1.954,4:1.628,5:1.427,6:1.287,7:1.182,8:1.099,9:1.032,10:0.975}
    return T.get(n,3/(c4a(n)*n**0.5))
def D3(n):
    T={2:0,3:0,4:0,5:0,6:0,7:0.076,8:0.136,9:0.184,10:0.223}; return T.get(n,0)
def D4(n):
    T={2:3.267,3:2.574,4:2.282,5:2.114,6:2.004,7:1.924,8:1.864,9:1.816,10:1.777}
    return T.get(n,2+3/d2(n))
def B3(n):
    T={2:0,3:0,4:0,5:0,6:0.030,7:0.118,8:0.185,9:0.239,10:0.284}; return T.get(n,0)
def B4(n):
    T={2:3.267,3:2.568,4:2.266,5:2.089,6:1.970,7:1.882,8:1.815,9:1.761,10:1.716}
    return T.get(n,1+3/(c4a(n)*(2*(n-1))**0.5))

def capability(data, usl, lsl, n_sub=1):
    mu=float(np.mean(data)); s_o=float(np.std(data,ddof=1))
    if n_sub==1:
        MR=np.abs(np.diff(data)); s_w=float(np.mean(MR)/d2(2))
    else:
        ng=len(data)//n_sub; gs=[data[i*n_sub:(i+1)*n_sub] for i in range(ng)]
        s_w=float(np.mean([np.max(g)-np.min(g) for g in gs])/d2(n_sub))
    tol=usl-lsl
    Cp =tol/(6*s_w)  if s_w>0 else np.nan
    Cpk=min((usl-mu)/(3*s_w),(mu-lsl)/(3*s_w)) if s_w>0 else np.nan
    Pp =tol/(6*s_o)  if s_o>0 else np.nan
    Ppk=min((usl-mu)/(3*s_o),(mu-lsl)/(3*s_o)) if s_o>0 else np.nan
    z_u=(usl-mu)/s_o if s_o>0 else 99; z_l=(mu-lsl)/s_o if s_o>0 else 99
    ppm=(norm.sf(z_u)+norm.cdf(-z_l))*1e6
    return dict(mu=mu,s_w=s_w,s_o=s_o,Cp=Cp,Cpk=Cpk,Pp=Pp,Ppk=Ppk,
                ppm=ppm,yld=(1-ppm/1e6)*100)

def nelson(data, cl, ucl, lcl):
    n=len(data); v={}
    r1=[i for i,x in enumerate(data) if x>ucl or x<lcl]
    if r1: v["Rule 1 · ±3σ 관리한계 이탈"]=r1
    r2=[i for i in range(8,n) if all(x>cl for x in data[i-8:i+1]) or all(x<cl for x in data[i-8:i+1])]
    if r2: v["Rule 2 · 연속 9점 중심선 편향"]=r2
    r3=[]
    for i in range(5,n):
        w=data[i-5:i+1]; df=[w[j+1]-w[j] for j in range(5)]
        if all(x>0 for x in df) or all(x<0 for x in df): r3.append(i)
    if r3: v["Rule 3 · 연속 6점 단조 추세"]=r3
    r4=[i for i in range(13,n) if all((data[j]-data[j-1])*(data[j+1]-data[j])<0 for j in range(i-12,i))]
    if r4: v["Rule 4 · 연속 14점 상하 진동"]=r4
    return v

def grade(cpk):
    if np.isnan(cpk): return "N/A","b-warn","#d97706"
    if cpk>=1.67: return "우수 ★★★","b-good","#059669"
    if cpk>=1.33: return "양호 ★★","b-good","#10b981"
    if cpk>=1.00: return "보통 ★","b-warn","#f59e0b"
    return "불량","b-bad","#e11d48"

# Plotly light theme
PL = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(248,250,252,0.8)',
    font=dict(family='DM Sans', color='#64748b', size=11),
    xaxis=dict(gridcolor='rgba(226,232,240,0.8)', linecolor='#e2e8f0',
               zerolinecolor='#e2e8f0', showgrid=True),
    yaxis=dict(gridcolor='rgba(226,232,240,0.8)', linecolor='#e2e8f0',
               zerolinecolor='#e2e8f0', showgrid=True),
    margin=dict(l=52,r=24,t=40,b=48),
    legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#e2e8f0', borderwidth=1),
)
def apl(fig,**kw):
    # Merge PL base layout with kw; kw values override PL to avoid duplicate-key TypeError
    merged = {**PL, **kw}
    fig.update_layout(**merged); return fig


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <div style="font-size:1.6rem;margin-bottom:6px;">⚙️</div>
      <div class="sb-brand-name">ProcessIQ</div>
      <div class="sb-brand-sub">SPC · 공정능력분석 플랫폼</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section">데이터 입력 방식</div>', unsafe_allow_html=True)
    mode = st.radio("", ["📊 샘플 데이터", "✏️ 직접 입력", "📁 CSV 업로드"],
                    label_visibility="collapsed")

    st.markdown('<div class="sb-section">규격 (Specification)</div>', unsafe_allow_html=True)
    usl = st.number_input("규격 상한 USL", value=110.0, step=0.5, format="%.2f")
    lsl = st.number_input("규격 하한 LSL", value=90.0,  step=0.5, format="%.2f")
    tgt = st.number_input("목표값 Target",  value=(usl+lsl)/2, step=0.5, format="%.2f")

    st.markdown('<div class="sb-section">관리도 설정</div>', unsafe_allow_html=True)
    nsub = st.selectbox("부분군 크기 n", [1,2,3,4,5,6,7,8,9,10], index=0)
    chart_sel = st.selectbox("계량형 관리도", ["I-MR (n=1)","X̄-R (n≤10)","X̄-S (n≥10)"])
    attr_sel  = st.selectbox("계수형 관리도", ["P 관리도","NP 관리도","C 관리도","U 관리도"])

    st.markdown("""
    <div class="sb-tip">
      💡 <strong style="color:#c7d2fe">사용 팁</strong><br>
      시나리오를 바꾸면 모든 분석이 즉시 갱신됩니다. <em>드리프트 공정</em>을 선택하면
      Nelson Rule 3 (추세) 감지를 실시간으로 확인할 수 있습니다.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:32px;font-size:0.65rem;color:rgba(255,255,255,0.2);text-align:center">스마트제조 과제 · 산업공학<br>© 2025 ProcessIQ</div>', unsafe_allow_html=True)


# ── DATA ──────────────────────────────────────────────────────────────────────
@st.cache_data
def make_sample(sc):
    np.random.seed(42); n=120

    def inject_outliers(base, positions, magnitudes):
        """지정 위치에 이상치 주입"""
        d = base.copy()
        for pos, mag in zip(positions, magnitudes):
            d[pos] += mag
        return d

    if sc == "정상 공정 (Cpk≈1.33)":
        # 안정적 공정이지만 산발적 이상치 5개 포함
        base = np.random.normal(100, 3, n)
        return inject_outliers(base,
            positions=[8, 27, 53, 81, 99],
            magnitudes=[+12, -11, +13, -14, +10])

    if sc == "평균 이동 (비센터링)":
        # 공정 평균 이동 + 이상치 7개 (상하 혼재)
        base = np.concatenate([np.random.normal(100,3,60), np.random.normal(106,3,60)])
        return inject_outliers(base,
            positions=[5, 22, 40, 61, 75, 90, 110],
            magnitudes=[+15, -13, +11, -16, +14, -12, +18])

    if sc == "드리프트 (추세 존재)":
        # 완만한 상승 추세 + 이상치 6개
        base = np.random.normal(100, 2, n) + np.linspace(0, 10, n)
        return inject_outliers(base,
            positions=[10, 30, 48, 65, 88, 105],
            magnitudes=[+14, -12, +16, +13, -15, +11])

    if sc == "과분산 (Cpk<1.0)":
        # 분산 매우 큼 + 이상치 8개 (극단값)
        base = np.random.normal(100, 6, n)
        return inject_outliers(base,
            positions=[3, 18, 35, 52, 68, 82, 97, 113],
            magnitudes=[+22, -20, +24, -18, +25, -22, +19, -21])

    if sc == "주기적 이상 공정":
        # 주기적 패턴 + 규칙적 이상치 (생산 교대 시점)
        t = np.arange(n)
        base = np.random.normal(100, 2, n) + 3*np.sin(2*np.pi*t/20)
        return inject_outliers(base,
            positions=[20, 40, 60, 80, 100, 115],
            magnitudes=[+16, -14, +18, -15, +17, -13])

    # 기본 fallback
    return np.random.normal(100, 3, n)

scenarios=[
    "정상 공정 (Cpk≈1.33)",
    "평균 이동 (비센터링)",
    "드리프트 (추세 존재)",
    "과분산 (Cpk<1.0)",
    "주기적 이상 공정",
]

if mode=="📊 샘플 데이터":
    sc=st.selectbox("시나리오 선택",scenarios)
    raw=make_sample(sc)
elif mode=="✏️ 직접 입력":
    txt=st.text_area("숫자 입력 (쉼표 또는 줄바꿈)",
        "100.2,99.8,101.1,98.9,100.5,99.3,101.8,98.7,100.9,99.1,101.3,98.5,"
        "100.7,99.6,101.2,98.8,100.4,99.7,101.5,98.6,100.1,99.4,101.6,98.4,100.8",
        height=130)
    try: raw=np.array([float(x.strip()) for x in txt.replace('\n',',').split(',') if x.strip()])
    except: st.error("숫자 형식을 확인하세요"); raw=make_sample("정상 공정 (Cpk≈1.33)")
else:
    up=st.file_uploader("CSV 파일",type=["csv"])
    if up:
        df_u=pd.read_csv(up); nc=df_u.select_dtypes(include=np.number).columns.tolist()
        col=st.selectbox("분석할 열",nc); raw=df_u[col].dropna().values
    else:
        st.info("CSV를 업로드하거나 샘플 데이터를 사용하세요")
        raw=make_sample("정상 공정 (Cpk≈1.33)")

data=raw.copy()
res=capability(data,usl,lsl,nsub)
mu,s_w,s_o=res['mu'],res['s_w'],res['s_o']
Cp,Cpk,Pp,Ppk=res['Cp'],res['Cpk'],res['Pp'],res['Ppk']
grd,gcls,gcol=grade(Cpk)
sw_stat,sw_p=stats.shapiro(data[:50] if len(data)>50 else data)
is_normal=sw_p>0.05


# ── PAGE HEADER ───────────────────────────────────────────────────────────────
if Cpk>=1.33: s_bg,s_txt,s_dot="#ecfdf5","#065f46","#10b981"
elif Cpk>=1.0: s_bg,s_txt,s_dot="#fffbeb","#92400e","#f59e0b"
else: s_bg,s_txt,s_dot="#fff1f2","#9f1239","#f43f5e"

st.markdown(f"""
<div class="page-header">
  <div>
    <div class="page-tag">Smart Manufacturing · SPC Dashboard</div>
    <h1 class="page-title">공정능력분석<br>& 통계적공정관리</h1>
    <p class="page-sub">
      데이터 <strong style="color:#1a1a2e">{len(data)}</strong>개 &nbsp;·&nbsp;
      USL <strong style="color:#1a1a2e">{usl:.1f}</strong> &nbsp;·&nbsp;
      LSL <strong style="color:#1a1a2e">{lsl:.1f}</strong> &nbsp;·&nbsp;
      공차 <strong style="color:#1a1a2e">{usl-lsl:.1f}</strong>
    </p>
  </div>
  <div class="status-pill" style="background:{s_bg};color:{s_txt};border:1.5px solid {s_dot}40;">
    <div class="status-dot" style="background:{s_dot};box-shadow:0 0 6px {s_dot}"></div>
    공정 상태 : <strong>{grd}</strong>
  </div>
</div>
""", unsafe_allow_html=True)


# ── METRIC STRIP ──────────────────────────────────────────────────────────────
def mc(label,val,sub,cls):
    return f'<div class="metric-card {cls}"><div class="mc-label">{label}</div><div class="mc-val">{val}</div><div class="mc-sub">{sub}</div></div>'

cpk_cls = "mc-emerald" if Cpk>=1.33 else "mc-amber" if Cpk>=1.0 else "mc-rose"
ppm_cls = "mc-emerald" if res['ppm']<3400 else "mc-amber" if res['ppm']<66800 else "mc-rose"

st.markdown(f"""
<div class="metric-strip">
  {mc("Cp · 단기 공정능력",f"{Cp:.3f}","치우침 미고려","mc-indigo")}
  {mc("Cpk · 단기 공정능력",f"{Cpk:.3f}","치우침 반영",cpk_cls)}
  {mc("Pp · 장기 공정성능",f"{Pp:.3f}","치우침 미고려","mc-violet")}
  {mc("Ppk · 장기 공정성능",f"{Ppk:.3f}","치우침 반영","mc-teal")}
</div>
<div class="metric-strip">
  {mc("공정 평균 μ",f"{mu:.4f}",f"Target {tgt:.2f}  편차 {mu-tgt:+.4f}","mc-slate")}
  {mc("단기 σ (within)",f"{s_w:.4f}","군내변동 기반","mc-sky")}
  {mc("장기 σ (overall)",f"{s_o:.4f}","전체변동 기반","mc-violet")}
  {mc("PPM 불량률",f"{res['ppm']:.0f}",f"수율 {res['yld']:.3f}%",ppm_cls)}
</div>
""", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
t1,t2,t3,t4,t5 = st.tabs([
    "📊 공정능력분석",
    "📈 계량형 관리도",
    "🔢 계수형 관리도",
    "🔍 이상탐지",
    "📋 종합 리포트"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    c1,c2 = st.columns([3,2], gap="large")

    with c1:
        st.markdown('<div class="sec">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">공정 분포 & 규격 분석 <span class="sec-badge sb-blue">Distribution</span></div>', unsafe_allow_html=True)

        x_min=min(data.min(),lsl-4*s_o); x_max=max(data.max(),usl+4*s_o)
        xr=np.linspace(x_min,x_max,400); yp=norm.pdf(xr,mu,s_o)

        fig=go.Figure()
        # 구간 채우기
        for mask,fc,nm in [
            (xr<lsl,"rgba(244,63,94,0.12)","LSL 미달 불량"),
            (xr>usl,"rgba(244,63,94,0.12)","USL 초과 불량"),
            ((xr>=lsl)&(xr<=usl),"rgba(16,185,129,0.08)","양품 영역"),
        ]:
            xm=xr[mask]; ym=norm.pdf(xm,mu,s_o)
            if len(xm):
                fig.add_trace(go.Scatter(
                    x=np.concatenate([[xm[0]],xm,[xm[-1]]]),
                    y=np.concatenate([[0],ym,[0]]),
                    fill='toself',fillcolor=fc,line=dict(width=0),name=nm,hoverinfo='skip'))
        # ±3σ 음영
        m3=(xr>=mu-3*s_o)&(xr<=mu+3*s_o)
        fig.add_trace(go.Scatter(x=xr[m3],y=yp[m3],fill='tozeroy',
            fillcolor='rgba(99,102,241,0.06)',line=dict(width=0),name='±3σ',hoverinfo='skip'))
        # PDF
        fig.add_trace(go.Scatter(x=xr,y=yp,mode='lines',
            line=dict(color='#6366f1',width=2.5),name='정규 분포'))
        # 히스토그램
        fig.add_trace(go.Histogram(x=data,nbinsx=26,histnorm='probability density',
            marker=dict(color='rgba(99,102,241,0.2)',line=dict(color='rgba(99,102,241,0.5)',width=0.5)),
            name='실제 데이터'))
        # 수직선
        for val,clr,lbl,dash in [
            (usl,'#e11d48',f'USL={usl:.1f}','dash'),
            (lsl,'#e11d48',f'LSL={lsl:.1f}','dash'),
            (mu,'#6366f1',f'μ={mu:.3f}','solid'),
            (tgt,'#f59e0b',f'Target={tgt:.1f}','dot'),
            (mu+3*s_o,'#94a3b8','μ+3σ','dot'),
            (mu-3*s_o,'#94a3b8','μ-3σ','dot'),
        ]:
            fig.add_vline(x=val,line_dash=dash,line_color=clr,line_width=1.5,
                annotation_text=lbl,annotation_font_color=clr,annotation_font_size=10)

        apl(fig,height=380,xaxis_title="측정값",yaxis_title="확률 밀도",
            legend=dict(orientation='h',y=-0.22,font=dict(size=10),bgcolor='rgba(0,0,0,0)',borderwidth=0))
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

        st.markdown(f"""
        <div class="formula-wrap">
        Cp  = (USL − LSL) / 6σ_w = ({usl:.1f} − {lsl:.1f}) / (6 × {s_w:.4f}) = <strong>{Cp:.4f}</strong><br>
        Cpk = min[(USL−μ)/3σ_w, (μ−LSL)/3σ_w]<br>
            = min[{(usl-mu)/(3*s_w):.4f}, {(mu-lsl)/(3*s_w):.4f}] = <strong>{Cpk:.4f}</strong><br><br>
        Pp  = (USL − LSL) / 6σ_o = ({usl:.1f} − {lsl:.1f}) / (6 × {s_o:.4f}) = <strong>{Pp:.4f}</strong><br>
        Ppk = min[{(usl-mu)/(3*s_o):.4f}, {(mu-lsl)/(3*s_o):.4f}] = <strong>{Ppk:.4f}</strong>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        # Gauge
        st.markdown('<div class="sec">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Cpk 게이지 <span class="sec-badge sb-green">Capability</span></div>', unsafe_allow_html=True)
        fig_g=go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=float(Cpk),
            number=dict(font=dict(size=38,color='#1a1a2e',family='JetBrains Mono'),valueformat='.3f'),
            delta=dict(reference=1.33,valueformat='.3f',
                       increasing=dict(color='#059669'),decreasing=dict(color='#e11d48')),
            gauge=dict(
                axis=dict(range=[0,2.5],tickcolor='#cbd5e1',tickfont=dict(color='#94a3b8',size=10)),
                bar=dict(color='#6366f1',thickness=0.2),
                bgcolor='rgba(241,245,249,0.8)', borderwidth=0,
                steps=[
                    dict(range=[0,1.0],   color='rgba(244,63,94,0.15)'),
                    dict(range=[1.0,1.33], color='rgba(245,158,11,0.15)'),
                    dict(range=[1.33,1.67],color='rgba(16,185,129,0.12)'),
                    dict(range=[1.67,2.5], color='rgba(16,185,129,0.06)'),
                ],
                threshold=dict(line=dict(color='#f59e0b',width=2.5),thickness=0.75,value=1.33)),
            title=dict(text="Cpk · 단기 공정능력지수",font=dict(color='#64748b',size=12))
        ))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='DM Sans',color='#64748b'),
            height=260,margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(fig_g,use_container_width=True,config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Grade table
        st.markdown(f"""
        <div class="sec" style="margin-top:14px">
          <div class="sec-title">판정 기준표</div>
          <table class="stbl">
            <tr><td>Cpk ≥ 1.67</td><td><span class="badge b-good">우수 ★★★</span></td></tr>
            <tr><td>1.33 ≤ Cpk &lt; 1.67</td><td><span class="badge b-good">양호 ★★</span></td></tr>
            <tr><td>1.00 ≤ Cpk &lt; 1.33</td><td><span class="badge b-warn">보통 ★</span></td></tr>
            <tr><td>Cpk &lt; 1.00</td><td><span class="badge b-bad">불량</span></td></tr>
          </table>
          <hr class="divider" style="margin:14px 0">
          <div style="display:flex;align-items:center;justify-content:space-between">
            <span style="font-size:0.8rem;color:#64748b">현재 판정</span>
            <span class="badge {gcls}">{grd}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Cp vs Cpk
        st.markdown('<div class="sec" style="margin-top:14px"><div class="sec-title">능력지수 비교</div>', unsafe_allow_html=True)
        idxs=['Cp','Cpk','Pp','Ppk']; vals=[Cp,Cpk,Pp,Ppk]
        bcs=[('#059669' if v>=1.33 else '#f59e0b' if v>=1.0 else '#e11d48') for v in vals]
        fig_b=go.Figure(go.Bar(x=idxs,y=vals,marker_color=bcs,marker_line_width=0,width=0.55,
            text=[f'{v:.3f}' for v in vals],textposition='outside',
            textfont=dict(color='#1a1a2e',size=12,family='JetBrains Mono')))
        for ref,clr,lbl in [(1.33,'#059669','권고 1.33'),(1.0,'#f59e0b','최소 1.00')]:
            fig_b.add_hline(y=ref,line_dash='dash',line_color=clr,line_width=1.2,
                annotation_text=lbl,annotation_font_color=clr,annotation_font_size=9)
        apl(fig_b,height=220,yaxis_range=[0,max(vals+[1.9])*1.18],showlegend=False,
            margin=dict(l=36,r=16,t=12,b=28))
        st.plotly_chart(fig_b,use_container_width=True,config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Normality
    st.markdown('<div class="sec"><div class="sec-title">정규성 검정 (Shapiro-Wilk) & Q-Q Plot <span class="sec-badge sb-violet">Normality</span></div>', unsafe_allow_html=True)
    cn1,cn2=st.columns([1,2])
    with cn1:
        sw_c='#059669' if is_normal else '#e11d48'
        st.markdown(f"""
        <div style="text-align:center;padding:16px 0">
          <div style="font-family:'JetBrains Mono';font-size:2.2rem;font-weight:500;color:{sw_c}">{sw_stat:.4f}</div>
          <div style="font-size:0.75rem;color:#94a3b8;margin:4px 0 10px">W 통계량 · p = {sw_p:.4f}</div>
          <span class="badge {'b-good' if is_normal else 'b-bad'}">{'✓ 정규분포 성립' if is_normal else '✗ 비정규 분포'}</span>
          <div class="{'good-b' if is_normal else 'warn-b'}" style="margin-top:12px;text-align:left">
            {'Cp, Cpk 지수 신뢰 가능합니다.' if is_normal else 'Box-Cox 또는 Johnson 변환 적용을 권장합니다.'}
          </div>
        </div>
        """, unsafe_allow_html=True)
    with cn2:
        (osm,osr),(slope,intercept,r2)=stats.probplot(data)
        qt,qs=np.array(osm),np.array(osr); lx=np.linspace(qt.min(),qt.max(),100)
        fig_qq=go.Figure()
        fig_qq.add_trace(go.Scatter(x=qt,y=qs,mode='markers',name='관측치',
            marker=dict(color='#6366f1',size=5,opacity=0.7)))
        fig_qq.add_trace(go.Scatter(x=lx,y=slope*lx+intercept,mode='lines',
            name=f'정규선 (R²={r2**2:.4f})',line=dict(color='#e11d48',width=1.5,dash='dash')))
        apl(fig_qq,height=240,xaxis_title="이론적 분위수",yaxis_title="표본 분위수")
        st.plotly_chart(fig_qq,use_container_width=True,config={'displayModeBar':False})
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    if "I-MR" in chart_sel or nsub==1:
        MR=np.abs(np.diff(data)); Xb=float(np.mean(data)); MRb=float(np.mean(MR))
        I_ucl=Xb+2.660*MRb; I_lcl=Xb-2.660*MRb; MR_ucl=3.267*MRb

        vI=nelson(data,Xb,I_ucl,I_lcl)
        allV=set(i for v in vI.values() for i in v)
        ci=[('#e11d48' if i in allV else '#6366f1') for i in range(len(data))]
        cMR=[('#e11d48' if i in set(i2 for v in nelson(MR,MRb,MR_ucl,0).values() for i2 in v) else '#14b8a6')
             for i in range(len(MR))]

        st.markdown('<div class="sec"><div class="sec-title">I-MR 관리도 <span class="sec-badge sb-blue">개별값 · 이동범위</span></div>', unsafe_allow_html=True)
        fig2=make_subplots(rows=2,cols=1,
            subplot_titles=("I 관리도 — 개별측정치 (Individual)","MR 관리도 — 이동범위 (Moving Range)"),
            shared_xaxes=True,vertical_spacing=0.14)

        fig2.add_trace(go.Scatter(y=data,mode='lines+markers',
            line=dict(color='rgba(99,102,241,0.4)',width=1.2),
            marker=dict(color=ci,size=5.5,line=dict(width=0.5,color='rgba(0,0,0,0.1)')),
            name='개별값 (I)'),row=1,col=1)
        for val,clr,dash,lbl in [(I_ucl,'#e11d48','dash',f'UCL={I_ucl:.4f}'),
                                   (Xb,'#059669','solid',f'CL={Xb:.4f}'),
                                   (I_lcl,'#e11d48','dash',f'LCL={I_lcl:.4f}')]:
            fig2.add_hline(y=val,line_dash=dash,line_color=clr,line_width=1.5,
                annotation_text=lbl,annotation_font_color=clr,annotation_font_size=9,row=1,col=1)
        for i in allV:
            fig2.add_vrect(x0=i-0.5,x1=i+0.5,fillcolor='rgba(244,63,94,0.08)',
                line_width=0,row=1,col=1)

        fig2.add_trace(go.Scatter(y=MR,mode='lines+markers',
            line=dict(color='rgba(20,184,166,0.4)',width=1.2),
            marker=dict(color=cMR,size=5.5),name='이동범위 (MR)'),row=2,col=1)
        for val,clr,dash,lbl in [(MR_ucl,'#e11d48','dash',f'UCL={MR_ucl:.4f}'),
                                   (MRb,'#059669','solid',f'CL={MRb:.4f}'),
                                   (0,'#94a3b8','dot','LCL=0')]:
            fig2.add_hline(y=val,line_dash=dash,line_color=clr,line_width=1.5,
                annotation_text=lbl,annotation_font_color=clr,annotation_font_size=9,row=2,col=1)

        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,250,252,0.8)',
            font=dict(family='DM Sans',color='#64748b',size=10),
            height=540,showlegend=False,margin=dict(l=52,r=24,t=52,b=40))
        fig2.update_xaxes(gridcolor='rgba(226,232,240,0.8)',linecolor='#e2e8f0')
        fig2.update_yaxes(gridcolor='rgba(226,232,240,0.8)',linecolor='#e2e8f0')
        st.plotly_chart(fig2,use_container_width=True,config={'displayModeBar':False})

        st.markdown(f"""
        <div class="formula-wrap">
        I 관리도 : UCL = X̄ + E₂·MR̄ = {Xb:.4f} + 2.660 × {MRb:.4f} = {I_ucl:.4f}<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;LCL = X̄ − E₂·MR̄ = {Xb:.4f} − 2.660 × {MRb:.4f} = {I_lcl:.4f}<br>
        MR 관리도: UCL = D₄·MR̄ = 3.267 × {MRb:.4f} = {MR_ucl:.4f}  |  LCL = 0<br>
        σ_within = MR̄ / d₂(2) = {MRb:.4f} / 1.128 = {s_w:.4f}
        </div>
        """, unsafe_allow_html=True)

        # Nelson violations
        total_v=sum(len(v) for v in vI.values())
        cv1,cv2=st.columns([1,2])
        with cv1:
            vc='#059669' if total_v==0 else '#e11d48'
            vt='관리상태 (In Control)' if total_v==0 else f'이상 {total_v}건 감지'
            st.markdown(f"""
            <div style="text-align:center;padding:20px;background:{'#ecfdf5' if total_v==0 else '#fff1f2'};border-radius:12px;border:1px solid {'#a7f3d0' if total_v==0 else '#fecdd3'}">
              <div style="font-size:2.4rem;line-height:1">{'✓' if total_v==0 else '✗'}</div>
              <div style="color:{vc};font-weight:600;margin:6px 0 2px;font-size:0.95rem">{vt}</div>
              <div style="font-size:0.72rem;color:#94a3b8">Nelson's Rules 4가지 기준</div>
            </div>""", unsafe_allow_html=True)
        with cv2:
            if vI:
                for rule,idxs in vI.items():
                    st.markdown(f"""
                    <div class="viol-item">
                      <div class="viol-title">{rule}</div>
                      <div class="viol-body">이상 위치 → #{', #'.join(str(i+1) for i in idxs[:12])}{'...' if len(idxs)>12 else ''}</div>
                    </div>""", unsafe_allow_html=True)
                st.markdown('<div class="warn-b">📌 이상원인 파악 후 해당 데이터를 제거하고 관리한계선을 재계산하세요. 모든 점이 관리상태가 될 때까지 반복합니다.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="good-b">✅ 4가지 Nelson\'s Rules 모두 통과 — 공정이 안정된 관리상태입니다.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        ng=len(data)//nsub; gs=[data[i*nsub:(i+1)*nsub] for i in range(ng)]
        gm=np.array([np.mean(g) for g in gs]); gR=np.array([np.max(g)-np.min(g) for g in gs])
        gS=np.array([np.std(g,ddof=1) for g in gs])
        Xdb=float(np.mean(gm)); Rb=float(np.mean(gR)); Sb=float(np.mean(gS))
        useR="X̄-R" in chart_sel

        if useR:
            a=A2(nsub); Xu=Xdb+a*Rb; Xl=Xdb-a*Rb
            su=D4(nsub)*Rb; sl=D3(nsub)*Rb; sc2=Rb; sd=gR; slb="R (범위)"
        else:
            a=A3(nsub); Xu=Xdb+a*Sb; Xl=Xdb-a*Sb
            su=B4(nsub)*Sb; sl=B3(nsub)*Sb; sc2=Sb; sd=gS; slb="S (표준편차)"

        vX=nelson(gm,Xdb,Xu,Xl); allVx=set(i for v in vX.values() for i in v)
        cx=[('#e11d48' if i in allVx else '#6366f1') for i in range(len(gm))]

        st.markdown('<div class="sec">', unsafe_allow_html=True)
        fig_xr=make_subplots(rows=2,cols=1,
            subplot_titles=(f"X̄ 관리도 — 부분군 평균 (n={nsub})",f"{'R' if useR else 'S'} 관리도 — 산포"),
            shared_xaxes=True,vertical_spacing=0.14)
        fig_xr.add_trace(go.Scatter(y=gm,mode='lines+markers',
            line=dict(color='rgba(99,102,241,0.4)',width=1.5),
            marker=dict(color=cx,size=7),name='X̄'),row=1,col=1)
        for val,clr,dash,lbl in [(Xu,'#e11d48','dash',f'UCL={Xu:.4f}'),
                                   (Xdb,'#059669','solid',f'X̄̄={Xdb:.4f}'),
                                   (Xl,'#e11d48','dash',f'LCL={Xl:.4f}')]:
            fig_xr.add_hline(y=val,line_dash=dash,line_color=clr,line_width=1.5,
                annotation_text=lbl,annotation_font_color=clr,annotation_font_size=9,row=1,col=1)
        fig_xr.add_trace(go.Scatter(y=sd,mode='lines+markers',
            line=dict(color='rgba(139,92,246,0.4)',width=1.5),
            marker=dict(color='#8b5cf6',size=6),name=slb),row=2,col=1)
        for val,clr,dash,lbl in [(su,'#e11d48','dash',f'UCL={su:.4f}'),
                                   (sc2,'#059669','solid',f'CL={sc2:.4f}'),
                                   (sl,'#e11d48','dash',f'LCL={sl:.4f}')]:
            fig_xr.add_hline(y=val,line_dash=dash,line_color=clr,line_width=1.5,
                annotation_text=lbl,annotation_font_color=clr,annotation_font_size=9,row=2,col=1)
        fig_xr.update_layout(paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,250,252,0.8)',
            font=dict(family='DM Sans',color='#64748b',size=10),
            height=540,showlegend=False,margin=dict(l=52,r=24,t=52,b=40))
        fig_xr.update_xaxes(gridcolor='rgba(226,232,240,0.8)',linecolor='#e2e8f0')
        fig_xr.update_yaxes(gridcolor='rgba(226,232,240,0.8)',linecolor='#e2e8f0')
        st.plotly_chart(fig_xr,use_container_width=True,config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── CUSUM + EWMA (계량형 관리도 탭 공통) ──────────────────────────────────
    st.markdown('<div style="margin:8px 0 4px;font-size:0.72rem;font-weight:700;color:#94a3b8;letter-spacing:1.5px;text-transform:uppercase">▸ 시간가중형 관리도 (Time-Weighted Charts)</div>', unsafe_allow_html=True)
    _lam2 = 0.20
    _k2 = 0.5 * s_w
    _cp2 = np.zeros(len(data)); _cn2 = np.zeros(len(data))
    _ew2 = np.zeros(len(data)); _ew2[0] = data[0]
    for _i in range(1, len(data)):
        _cp2[_i] = max(0, _cp2[_i-1] + (data[_i] - mu) - _k2)
        _cn2[_i] = min(0, _cn2[_i-1] + (data[_i] - mu) + _k2)
        _ew2[_i] = _lam2 * data[_i] + (1 - _lam2) * _ew2[_i-1]
    _H2 = 5 * s_w
    _ew_s2 = s_w * np.sqrt(_lam2 / (2 - _lam2))
    _ew_ucl2 = mu + 3 * _ew_s2; _ew_lcl2 = mu - 3 * _ew_s2

    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown('<div class="sec"><div class="sec-title">CUSUM 누적합 관리도 <span class="sec-badge sb-violet">시간가중형</span></div>', unsafe_allow_html=True)
        _fig_cs2 = go.Figure()
        _fig_cs2.add_trace(go.Scatter(y=_cp2, mode='lines', line=dict(color='#6366f1', width=1.8), name='C⁺ (상향)'))
        _fig_cs2.add_trace(go.Scatter(y=_cn2, mode='lines', line=dict(color='#e11d48', width=1.8), name='C⁻ (하향)'))
        _fig_cs2.add_hline(y=_H2, line_dash='dash', line_color='#f59e0b', line_width=1.3,
            annotation_text=f'H={_H2:.2f}', annotation_font_color='#d97706', annotation_font_size=9)
        _fig_cs2.add_hline(y=-_H2, line_dash='dash', line_color='#f59e0b', line_width=1.3)
        _fig_cs2.add_hline(y=0, line_color='#cbd5e1', line_width=0.6)
        # 이탈 구간 강조
        for _i, (cp_v, cn_v) in enumerate(zip(_cp2, _cn2)):
            if cp_v > _H2 or cn_v < -_H2:
                _fig_cs2.add_shape(type='line', x0=_i, x1=_i, y0=-_H2*1.1, y1=_H2*1.1,
                    line=dict(color='rgba(244,63,94,0.15)', width=6))
        apl(_fig_cs2, height=290, xaxis_title="관측 순서", yaxis_title="누적합 (CUSUM)")
        st.plotly_chart(_fig_cs2, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f'<div class="formula-wrap" style="font-size:0.73rem">C⁺ᵢ = max(0, Cᵢ₋₁⁺ + (xᵢ−μ) − k) &nbsp;·&nbsp; k = 0.5σ = {_k2:.4f}<br>C⁻ᵢ = min(0, Cᵢ₋₁⁻ + (xᵢ−μ) + k) &nbsp;·&nbsp; H = 5σ = {_H2:.4f}<br>H 초과 시 이상 신호 — 소규모 평균 이동 조기 탐지에 강함</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tc2:
        st.markdown('<div class="sec"><div class="sec-title">EWMA 지수가중이동평균 관리도 <span class="sec-badge sb-teal">시간가중형</span></div>', unsafe_allow_html=True)
        _ew_viol2 = [_i for _i, v in enumerate(_ew2) if v > _ew_ucl2 or v < _ew_lcl2]
        _fig_ew2 = go.Figure()
        _fig_ew2.add_trace(go.Scatter(y=data, mode='markers',
            marker=dict(color='rgba(99,102,241,0.2)', size=4), name='원시 데이터'))
        _fig_ew2.add_trace(go.Scatter(y=_ew2, mode='lines',
            line=dict(color='#14b8a6', width=2.2), name=f'EWMA (λ={_lam2})'))
        if _ew_viol2:
            _fig_ew2.add_trace(go.Scatter(x=_ew_viol2, y=[_ew2[_i] for _i in _ew_viol2],
                mode='markers', marker=dict(color='#e11d48', size=9, symbol='x',
                line=dict(color='#e11d48', width=2)), name=f'이탈 ({len(_ew_viol2)}건)'))
        for _val, _clr, _dash, _lbl in [
            (_ew_ucl2, '#e11d48', 'dash', f'UCL={_ew_ucl2:.4f}'),
            (mu,       '#059669', 'solid', f'CL={mu:.4f}'),
            (_ew_lcl2, '#e11d48', 'dash', f'LCL={_ew_lcl2:.4f}'),
        ]:
            _fig_ew2.add_hline(y=_val, line_dash=_dash, line_color=_clr, line_width=1.3,
                annotation_text=_lbl, annotation_font_color=_clr, annotation_font_size=9)
        apl(_fig_ew2, height=290, xaxis_title="관측 순서", yaxis_title="EWMA")
        st.plotly_chart(_fig_ew2, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f'<div class="formula-wrap" style="font-size:0.73rem">EWMAᵢ = λ·xᵢ + (1−λ)·EWMAᵢ₋₁ &nbsp;·&nbsp; λ = {_lam2}<br>UCL/LCL = μ ± 3σ√(λ/(2−λ)) = {mu:.4f} ± {3*_ew_s2:.4f}<br>지수 감소 가중치로 최근 데이터를 더 민감하게 반영</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.markdown('<div class="info-b">📌 계수형 관리도 — 이항분포 기반(불량 P/NP)과 포아송분포 기반(결점 C/U) 두 가지 유형을 지원합니다.</div>', unsafe_allow_html=True)
    ca1,ca2,ca3=st.columns(3)
    with ca1: na=int(st.number_input("부분군 수",value=25,min_value=5,max_value=100))
    with ca2: sn=int(st.number_input("부분군 크기 n",value=100,min_value=10))
    with ca3: asc=st.selectbox("시나리오",["정상 공정","불량률 점진 증가","산발적 이상"])

    np.random.seed(42)
    if asc=="정상 공정": dc=np.random.binomial(sn,0.04,na).astype(float)
    elif asc=="불량률 점진 증가":
        pv=np.linspace(0.02,0.12,na); dc=np.array([np.random.binomial(sn,p) for p in pv]).astype(float)
    else:
        dc=np.random.binomial(sn,0.04,na).astype(float); dc[4]=sn*0.18; dc[14]=sn*0.21

    dr=dc/sn; pb=float(np.mean(dr))
    if "P 관리도" in attr_sel:
        cl_a=pb; ucl_a=pb+3*np.sqrt(pb*(1-pb)/sn); lcl_a=max(0,pb-3*np.sqrt(pb*(1-pb)/sn))
        yd=dr; yl="불량률 (p)"; ttl=f"P 관리도 — 불량률 · p̄ = {pb:.4f}"
    elif "NP" in attr_sel:
        npb=float(np.mean(dc)); cl_a=npb; ucl_a=npb+3*np.sqrt(npb*(1-pb)); lcl_a=max(0,npb-3*np.sqrt(npb*(1-pb)))
        yd=dc; yl="불량수 (np)"; ttl=f"NP 관리도 — 불량수 · n·p̄ = {npb:.2f}"
    elif "C 관리도" in attr_sel:
        dpu=np.random.poisson(4,na).astype(float); cb=float(np.mean(dpu))
        cl_a=cb; ucl_a=cb+3*np.sqrt(cb); lcl_a=max(0,cb-3*np.sqrt(cb))
        yd=dpu; yl="결점수 (c)"; ttl=f"C 관리도 — 결점수 · c̄ = {cb:.2f}"
    else:
        dpu=np.random.poisson(4,na).astype(float)/sn; ub=float(np.mean(dpu))
        cl_a=ub; ucl_a=ub+3*np.sqrt(ub/sn); lcl_a=max(0,ub-3*np.sqrt(ub/sn))
        yd=dpu; yl="단위당 결점수 (u)"; ttl=f"U 관리도 — ū = {ub:.4f}"

    viol_a=[i for i,v in enumerate(yd) if v>ucl_a or v<lcl_a]
    cv_a=[('#e11d48' if i in viol_a else '#059669') for i in range(len(yd))]

    st.markdown('<div class="sec">', unsafe_allow_html=True)
    fig_a=go.Figure()
    fig_a.add_trace(go.Scatter(y=yd,mode='lines+markers',
        line=dict(color='rgba(5,150,105,0.4)',width=1.5),
        marker=dict(color=cv_a,size=7,line=dict(color='rgba(0,0,0,0.1)',width=0.5)),name=yl))
    for val,clr,dash,lbl in [(ucl_a,'#e11d48','dash',f'UCL={ucl_a:.4f}'),
                               (cl_a,'#059669','solid',f'CL={cl_a:.4f}'),
                               (lcl_a,'#e11d48','dash',f'LCL={lcl_a:.4f}')]:
        fig_a.add_hline(y=val,line_dash=dash,line_color=clr,line_width=1.5,
            annotation_text=lbl,annotation_font_color=clr,annotation_font_size=10)
    for i in viol_a:
        fig_a.add_annotation(x=i,y=yd[i],text="⚠",showarrow=True,arrowhead=2,
            arrowcolor='#e11d48',font=dict(color='#e11d48',size=14),ay=-28)
    apl(fig_a,height=380,title=dict(text=ttl,font=dict(color='#1a1a2e',size=13)),
        xaxis_title="부분군 번호",yaxis_title=yl)
    st.plotly_chart(fig_a,use_container_width=True,config={'displayModeBar':False})
    st.markdown('</div>', unsafe_allow_html=True)

    cs1,cs2,cs3=st.columns(3)
    for col_,val_,lbl_,sub_ in [
        (cs1,len(viol_a),"이탈점 수",f"/ {len(yd)} 부분군"),
        (cs2,f"{ucl_a-lcl_a:.4f}","관리한계 폭","UCL − LCL"),
        (cs3,f"{len(viol_a)/len(yd)*100:.1f}%","이탈률","전체 대비"),
    ]:
        with col_:
            st.markdown(f'<div class="metric-card mc-indigo"><div class="mc-label">{lbl_}</div><div class="mc-val">{val_}</div><div class="mc-sub">{sub_}</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    da1,da2=st.columns([4,1])
    with da2:
        ww=st.slider("이동평균 윈도우",3,15,5)
        zth=st.slider("Z-score 임계값",1.5,4.0,3.0,0.1)
        lam=st.slider("EWMA λ",0.05,0.40,0.20,0.05)

    ser=pd.Series(data); rm=ser.rolling(ww,center=True).mean(); rs=ser.rolling(ww,center=True).std()
    zs=((ser-rm)/rs).abs(); anom_idx=np.where(zs>zth)[0]

    with da1:
        st.markdown('<div class="sec"><div class="sec-title">시계열 이상탐지 — Z-score 기반 <span class="sec-badge sb-teal">Anomaly Detection</span></div>', unsafe_allow_html=True)
        fig_ts=go.Figure()
        xall=list(range(len(data)))
        for mult,alpha in [(2,0.07),(1,0.05)]:
            fig_ts.add_trace(go.Scatter(
                x=xall+xall[::-1],
                y=list(mu+mult*s_w*np.ones(len(data)))+list(mu-mult*s_w*np.ones(len(data)))[::-1],
                fill='toself',fillcolor=f'rgba(99,102,241,{alpha})',line=dict(width=0),
                hoverinfo='skip',name=f'±{mult}σ'))
        fig_ts.add_trace(go.Scatter(x=xall,y=rm,mode='lines',
            line=dict(color='rgba(245,158,11,0.8)',width=1.5,dash='dot'),name=f'이동평균({ww})'))
        fig_ts.add_trace(go.Scatter(x=xall,y=data,mode='lines+markers',
            line=dict(color='rgba(99,102,241,0.5)',width=1),
            marker=dict(color='#6366f1',size=4),name='공정 데이터'))
        if len(anom_idx):
            fig_ts.add_trace(go.Scatter(x=anom_idx,y=data[anom_idx],mode='markers',
                marker=dict(color='#e11d48',size=10,symbol='x',line=dict(color='#e11d48',width=2)),
                name=f'이상치 ({len(anom_idx)}건)'))
        for val,clr,lbl in [(mu+3*s_w,'#e11d48','μ+3σ'),(mu,'#059669','μ'),(mu-3*s_w,'#e11d48','μ-3σ')]:
            fig_ts.add_hline(y=val,line_dash='dash',line_color=clr,line_width=1,
                annotation_text=lbl,annotation_font_size=9,annotation_font_color=clr)
        apl(fig_ts,height=300,xaxis_title="관측 순서",yaxis_title="측정값")
        st.plotly_chart(fig_ts,use_container_width=True,config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    # CUSUM + EWMA
    k=0.5*s_w; cp2=np.zeros(len(data)); cn2=np.zeros(len(data))
    ew=np.zeros(len(data)); ew[0]=data[0]
    for i in range(1,len(data)):
        cp2[i]=max(0,cp2[i-1]+(data[i]-mu)-k); cn2[i]=min(0,cn2[i-1]+(data[i]-mu)+k)
        ew[i]=lam*data[i]+(1-lam)*ew[i-1]
    H=5*s_w; ew_s=s_w*np.sqrt(lam/(2-lam)); ew_ucl=mu+3*ew_s; ew_lcl=mu-3*ew_s

    cc1,cc2=st.columns(2)
    with cc1:
        st.markdown('<div class="sec"><div class="sec-title">CUSUM 누적합 관리도 <span class="sec-badge sb-violet">시간가중형</span></div>', unsafe_allow_html=True)
        fig_cs=go.Figure()
        fig_cs.add_trace(go.Scatter(y=cp2,mode='lines',line=dict(color='#6366f1',width=1.5),name='C+ (상향)'))
        fig_cs.add_trace(go.Scatter(y=cn2,mode='lines',line=dict(color='#e11d48',width=1.5),name='C− (하향)'))
        fig_cs.add_hline(y=H,line_dash='dash',line_color='#f59e0b',line_width=1.2,
            annotation_text=f'H={H:.2f}',annotation_font_color='#d97706')
        fig_cs.add_hline(y=-H,line_dash='dash',line_color='#f59e0b',line_width=1.2)
        fig_cs.add_hline(y=0,line_color='#cbd5e1',line_width=0.5)
        apl(fig_cs,height=270,xaxis_title="관측 순서",yaxis_title="누적합")
        st.plotly_chart(fig_cs,use_container_width=True,config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with cc2:
        st.markdown('<div class="sec"><div class="sec-title">EWMA 지수가중이동평균 관리도 <span class="sec-badge sb-teal">시간가중형</span></div>', unsafe_allow_html=True)
        ew_viol=[i for i,v in enumerate(ew) if v>ew_ucl or v<ew_lcl]
        fig_ew=go.Figure()
        fig_ew.add_trace(go.Scatter(y=data,mode='markers',
            marker=dict(color='rgba(99,102,241,0.2)',size=4),name='원시 데이터'))
        fig_ew.add_trace(go.Scatter(y=ew,mode='lines',
            line=dict(color='#14b8a6',width=2),name=f'EWMA (λ={lam})'))
        if ew_viol:
            fig_ew.add_trace(go.Scatter(x=ew_viol,y=[ew[i] for i in ew_viol],mode='markers',
                marker=dict(color='#e11d48',size=9,symbol='x'),name='이탈'))
        for val,clr,dash,lbl in [(ew_ucl,'#e11d48','dash',f'UCL={ew_ucl:.4f}'),
                                   (mu,'#059669','solid',f'CL={mu:.4f}'),
                                   (ew_lcl,'#e11d48','dash',f'LCL={ew_lcl:.4f}')]:
            fig_ew.add_hline(y=val,line_dash=dash,line_color=clr,line_width=1.2,
                annotation_text=lbl,annotation_font_color=clr,annotation_font_size=9)
        apl(fig_ew,height=270,xaxis_title="관측 순서",yaxis_title="EWMA")
        st.plotly_chart(fig_ew,use_container_width=True,config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Z-score heatmap
    st.markdown('<div class="sec"><div class="sec-title">이상치 심각도 히트맵 <span class="sec-badge sb-rose">Z-score Matrix</span></div>', unsafe_allow_html=True)
    zc=zs.fillna(0).values; chunk=10; nc2=len(zc)//chunk
    if nc2>0:
        zm=zc[:nc2*chunk].reshape(nc2,chunk)
        fig_h=go.Figure(go.Heatmap(z=zm,
            colorscale=[[0,'rgba(16,185,129,0.3)'],[0.5,'rgba(245,158,11,0.7)'],[1,'rgba(225,29,72,0.9)']],
            colorbar=dict(title='|Z|',tickfont=dict(color='#64748b',size=10)),
            text=[[f'{v:.2f}' for v in row] for row in zm],
            texttemplate='%{text}',textfont=dict(size=9,color='#1a1a2e'),zmin=0,zmax=4))
        apl(fig_h,height=230,xaxis_title="위치",yaxis_title="부분군")
        st.plotly_chart(fig_h,use_container_width=True,config={'displayModeBar':False})
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    rp1,rp2=st.columns(2,gap="large")
    with rp1:
        st.markdown(f"""
        <div class="sec">
          <div class="sec-title">공정 기본 정보 <span class="sec-badge sb-blue">Basic Stats</span></div>
          <table class="stbl">
            <tr><td>데이터 수</td><td>{len(data)}개</td></tr>
            <tr><td>공정 평균 (μ)</td><td>{mu:.4f}</td></tr>
            <tr><td>단기 σ (within)</td><td>{s_w:.4f}</td></tr>
            <tr><td>장기 σ (overall)</td><td>{s_o:.4f}</td></tr>
            <tr><td>σ 비율 (장기/단기)</td><td>{s_o/s_w:.4f}</td></tr>
            <tr><td>USL</td><td>{usl:.2f}</td></tr>
            <tr><td>LSL</td><td>{lsl:.2f}</td></tr>
            <tr><td>공차 (Tolerance)</td><td>{usl-lsl:.2f}</td></tr>
            <tr><td>목표값 (Target)</td><td>{tgt:.2f}</td></tr>
            <tr><td>평균 편차 |μ−Target|</td><td>{abs(mu-tgt):.4f}</td></tr>
            <tr><td>정규성 (Shapiro-Wilk p)</td><td style="color:{'#059669' if is_normal else '#e11d48'}">{sw_p:.4f} · {'통과' if is_normal else '기각'}</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    with rp2:
        def cv(v):
            c='#059669' if isinstance(v,float) and v>=1.33 else '#d97706' if isinstance(v,float) and v>=1.0 else '#e11d48' if isinstance(v,float) else '#1a1a2e'
            vstr = f'{v:.4f}' if isinstance(v, float) else str(v)
            return f'<td style="color:{c};font-family:JetBrains Mono,monospace;text-align:right;font-weight:500">{vstr}</td>'
        st.markdown(f"""
        <div class="sec">
          <div class="sec-title">능력지수 & 판정 <span class="sec-badge sb-green">Indices</span></div>
          <table class="stbl">
            <tr><td>Cp (단기·치우침 미고려)</td>{cv(Cp)}</tr>
            <tr><td>Cpk (단기·치우침 반영)</td>{cv(Cpk)}</tr>
            <tr><td>Pp (장기·치우침 미고려)</td>{cv(Pp)}</tr>
            <tr><td>Ppk (장기·치우침 반영)</td>{cv(Ppk)}</tr>
            <tr><td>PPM 불량률</td><td style="font-family:JetBrains Mono,monospace;text-align:right;color:{'#e11d48' if res['ppm']>3400 else '#059669'};font-weight:500">{res['ppm']:.1f}</td></tr>
            <tr><td>예상 수율</td><td style="font-family:JetBrains Mono,monospace;text-align:right;font-weight:500">{res['yld']:.4f}%</td></tr>
            <tr><td>종합 판정</td><td style="text-align:right"><span class="badge {gcls}">{grd}</span></td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    # Recommendations
    recs=[]
    if Cpk<1.0: recs.append(("#e11d48","#fff1f2","🔴 즉각 조치 필요",f"Cpk = {Cpk:.3f} < 1.00 — 공정이 규격을 충족하지 못합니다. 4M(사람·기계·재료·방법) 분석으로 이상원인을 파악하고 즉시 개선하세요."))
    elif Cpk<1.33: recs.append(("#d97706","#fffbeb","🟡 개선 권장",f"Cpk = {Cpk:.3f} — 최소 기준(1.00) 초과이나 권고 기준(1.33) 미달입니다. 공정 변동 감소를 위한 개선 활동이 필요합니다."))
    else: recs.append(("#059669","#ecfdf5","🟢 공정 양호",f"Cpk = {Cpk:.3f} ≥ 1.33 — 권고 기준 이상입니다. 현 상태를 유지하고 지속적 모니터링을 수행하세요."))
    if abs(mu-tgt)>0.3*s_w: recs.append(("#7c3aed","#f5f3ff","⚙️ 센터링 필요",f"μ = {mu:.3f}이 Target {tgt:.2f}에서 {abs(mu-tgt):.3f} 편차 — Cp({Cp:.3f}) > Cpk({Cpk:.3f}) 차이가 치우침을 의미합니다."))
    if s_o/s_w>1.25: recs.append(("#0d9488","#f0fdfa","📊 군간변동 주의",f"σ_overall/σ_within = {s_o/s_w:.3f} > 1.25 — 공구 마모, 재료 로트 변경 등 장기적 변동 요인을 점검하세요."))
    if not is_normal: recs.append(("#d97706","#fffbeb","📐 비정규성 경고",f"Shapiro-Wilk p = {sw_p:.4f} ≤ 0.05 — Box-Cox 또는 Johnson 변환 후 능력지수를 재산정하세요."))

    st.markdown(f'<div class="sec"><div class="sec-title">자동 개선 권고안 <span class="sec-badge sb-amber">Recommendations</span></div>', unsafe_allow_html=True)
    for clr,bg,title_r,body_r in recs:
        st.markdown(f"""
        <div class="rec-item" style="background:{bg};border-left-color:{clr}">
          <div class="rec-title" style="color:{clr}">{title_r}</div>
          <div class="rec-body">{body_r}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Radar
    st.markdown('<div class="sec"><div class="sec-title">공정 건강도 레이더 <span class="sec-badge sb-violet">Health Score</span></div>', unsafe_allow_html=True)
    cats=['Cp 능력','Cpk 능력','Pp 성능','Ppk 성능','정규성','센터링']
    sc3=[min(Cp/2.0,1.0)*100,min(Cpk/2.0,1.0)*100,min(Pp/2.0,1.0)*100,min(Ppk/2.0,1.0)*100,
         min(sw_p/0.05,1.0)*100 if sw_p<0.05 else 100,
         max(0,100-abs(mu-tgt)/(s_w+1e-9)*20)]
    fig_r=go.Figure()
    fig_r.add_trace(go.Scatterpolar(r=sc3+[sc3[0]],theta=cats+[cats[0]],
        fill='toself',fillcolor='rgba(99,102,241,0.1)',
        line=dict(color='#6366f1',width=2),name='현재 공정'))
    fig_r.add_trace(go.Scatterpolar(r=[66.5]*len(cats)+[66.5],theta=cats+[cats[0]],
        mode='lines',line=dict(color='#059669',width=1.2,dash='dash'),name='권고기준'))
    fig_r.update_layout(paper_bgcolor='rgba(0,0,0,0)',
        polar=dict(bgcolor='rgba(241,245,249,0.8)',
            radialaxis=dict(visible=True,range=[0,100],color='#94a3b8',
                gridcolor='rgba(226,232,240,0.8)',tickfont=dict(color='#94a3b8',size=9)),
            angularaxis=dict(color='#64748b',gridcolor='rgba(226,232,240,0.8)')),
        font=dict(family='DM Sans',color='#64748b',size=11),
        legend=dict(bgcolor='rgba(255,255,255,0.9)',bordercolor='#e2e8f0',borderwidth=1),
        height=400,margin=dict(l=60,r=60,t=30,b=30))
    st.plotly_chart(fig_r,use_container_width=True,config={'displayModeBar':False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Download
    rd={"항목":["데이터 수","공정 평균","단기 σ","장기 σ","USL","LSL","목표값","Cp","Cpk","Pp","Ppk","PPM","수율(%)","정규성 p","종합 판정"],
        "값":[len(data),f"{mu:.4f}",f"{s_w:.4f}",f"{s_o:.4f}",usl,lsl,tgt,
              f"{Cp:.4f}",f"{Cpk:.4f}",f"{Pp:.4f}",f"{Ppk:.4f}",f"{res['ppm']:.1f}",f"{res['yld']:.4f}",f"{sw_p:.4f}",grd]}
    df_r=pd.DataFrame(rd); buf=io.StringIO(); df_r.to_csv(buf,index=False,encoding='utf-8-sig')
    st.download_button("📥 분석 리포트 CSV 다운로드",
        data=buf.getvalue().encode('utf-8-sig'),
        file_name="ProcessIQ_Report.csv",mime="text/csv")