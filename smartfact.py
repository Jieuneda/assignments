"""
총괄생산계획 최적화 웹앱
- PuLP CBC Solver + Streamlit + Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pulp

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="총괄생산계획 최적화 | 원예장비",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #f4f7f4; }

    /* ── 사이드바 전체 ── */
    [data-testid="stSidebar"] { background: #0f2418 !important; }
    [data-testid="stSidebar"] section { background: #0f2418 !important; }

    /* 라벨 텍스트 */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small { color: #74c69d !important; font-size: 12px !important; }

    /* 섹션 헤더 */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #95d5b2 !important; font-size: 13px !important;
        text-transform: uppercase; letter-spacing: 0.5px; }

    /* 입력칸 — 모든 wrapper 동일 배경으로 통일 */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="base-input"],
    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] div[data-testid="stNumberInput"] > div {
        background: #1b4332 !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] input {
        color: #d8f3dc !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }

    /* ─ + 버튼 — 입력칸과 같은 배경, 아이콘만 밝게 */
    [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"],
    [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"] {
        background: #1b4332 !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"] svg,
    [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"] svg {
        fill: #95d5b2 !important;
        stroke: #95d5b2 !important;
    }
    [data-testid="stSidebar"] button p {
        color: #95d5b2 !important;
        font-weight: 700 !important;
    }

    /* 구분선 */
    [data-testid="stSidebar"] hr { border-color: #2d6a4f !important; }

    .kpi-box {
        background: white; border-radius: 14px;
        padding: 18px 16px 14px 16px; text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07); height: 110px;
    }
    .kpi-label { font-size: 11px; font-weight: 700; color: #888;
                 text-transform: uppercase; letter-spacing: 0.8px; }
    .kpi-value { font-size: 24px; font-weight: 800; margin-top: 6px; }
    .kpi-sub   { font-size: 11px; color: #aaa; margin-top: 4px; }

    .section-header {
        font-size: 17px; font-weight: 800; color: #1b5e20;
        border-bottom: 3px solid #a5d6a7; padding-bottom: 6px;
        margin: 24px 0 14px 0;
    }
    .chart-card {
        background: white; border-radius: 14px;
        padding: 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        margin-bottom: 14px;
    }
    .chart-title { font-size: 14px; font-weight: 700; color: #2e7d32; margin-bottom: 10px; }

    .strategy-box {
        border-radius: 12px; padding: 14px 20px; margin-bottom: 18px;
        display: flex; align-items: center; gap: 14px;
    }
    .badge {
        font-size: 13px; font-weight: 800; padding: 5px 14px;
        border-radius: 20px; white-space: nowrap;
    }
    .shadow-card {
        background: #f9fbe7; border-left: 4px solid #aed581;
        border-radius: 8px; padding: 12px 16px; margin: 6px 0;
        font-size: 13px;
    }
    .formula-box {
        background: #e8f5e9; border-left: 4px solid #66bb6a;
        border-radius: 8px; padding: 10px 14px;
        font-family: monospace; font-size: 12px; color: #1b5e20;
    }
    .tab-note {
        background: #fff8e1; border-left: 4px solid #ffca28;
        border-radius: 6px; padding: 8px 14px;
        font-size: 12px; color: #555; margin-bottom: 10px;
    }

</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 초기값
# ──────────────────────────────────────────────
DEFAULT_DEMAND = [1600, 3000, 3200, 3800, 2200, 2200]
MONTHS_6 = ["1월", "2월", "3월", "4월", "5월", "6월"]

# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:8px;'>
      <div style='width:36px; height:36px; background:#2d6a4f; border-radius:10px;
           display:flex; align-items:center; justify-content:center; font-size:20px;'>🌿</div>
      <div>
        <p style='color:#95d5b2 !important; font-weight:600; font-size:15px; margin:0;'>최적화 APP</p>
        <p style='color:#52796f !important; font-size:11px; margin:0;'>원예장비 제조업체</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='color:#95d5b2;font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;'>📅 월별 수요 (개)</p>", unsafe_allow_html=True)
    demand = []
    c1, c2 = st.columns(2)
    for i, m in enumerate(MONTHS_6):
        col = c1 if i % 2 == 0 else c2
        with col:
            d = st.number_input(m, min_value=0, max_value=20000,
                                value=DEFAULT_DEMAND[i], step=100, key=f"d{i}")
            demand.append(d)

    st.markdown("---")
    st.markdown("<p style='color:#95d5b2;font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;'>💰 비용 단가 (천원)</p>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        c_regular  = st.number_input("정규임금(원/Hr)", value=4,   step=1)
        c_overtime = st.number_input("초과임금(원/Hr)", value=6,   step=1)
        c_hire     = st.number_input("고용비(천원/인)", value=300, step=10)
        c_fire     = st.number_input("해고비(천원/인)", value=500, step=10)
    with cb:
        c_holding  = st.number_input("재고유지비(천원/개·월)", value=2,  step=1)
        c_shortage = st.number_input("부족비(천원/개·월)",    value=5,  step=1)
        c_material = st.number_input("재료비(천원/개)",       value=10, step=1)
        c_subcon   = st.number_input("하청비(천원/개)",       value=30, step=1)
        selling_p  = st.number_input("판매가(천원/개)",       value=40, step=1)

    st.markdown("---")
    st.markdown("<p style='color:#95d5b2;font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;'>🏭 생산 파라미터</p>", unsafe_allow_html=True)
    work_days    = st.number_input("작업일수(일/월)",      value=20, step=1)
    work_hours   = st.number_input("작업시간(Hr/일)",      value=8,  step=1)
    std_time     = st.number_input("작업표준시간(Hr/개)",  value=4,  step=1)
    max_ot_per_w = st.number_input("최대초과시간(Hr/인·월)", value=10, step=1)
    W0           = st.number_input("초기 인력 W₀(명)",    value=80, step=1)
    I0           = st.number_input("초기 재고 I₀(개)",    value=1000, step=100)
    I_final_min  = st.number_input("기말 최소재고(개)",   value=500, step=100)

cap_per_worker = (work_hours * work_days) / std_time  # 8×20/4 = 40

# 목적함수 계수
coeff_W = c_regular * work_hours * work_days  # 640
coeff_O = c_overtime   # 6
coeff_H = c_hire       # 300
coeff_L = c_fire       # 500
coeff_I = c_holding    # 2
coeff_S = c_shortage   # 5
coeff_P = c_material   # 10
coeff_C = c_subcon     # 30


# ──────────────────────────────────────────────
# 최적화 모델
# ──────────────────────────────────────────────
def solve_app(demand, coeff_W, coeff_O, coeff_H, coeff_L,
              coeff_I, coeff_S, coeff_P, coeff_C,
              cap_per_worker, max_ot_per_w, W0, I0, I_final_min):
    TH = len(demand)
    T  = range(1, TH + 1)

    prob = pulp.LpProblem("APP", pulp.LpMinimize)

    W = {t: pulp.LpVariable(f"W_{t}", lowBound=0) for t in range(TH + 1)}
    H = {t: pulp.LpVariable(f"H_{t}", lowBound=0) for t in T}
    L = {t: pulp.LpVariable(f"L_{t}", lowBound=0) for t in T}
    P = {t: pulp.LpVariable(f"P_{t}", lowBound=0) for t in T}
    I = {t: pulp.LpVariable(f"I_{t}", lowBound=0) for t in range(TH + 1)}
    S = {t: pulp.LpVariable(f"S_{t}", lowBound=0) for t in range(TH + 1)}
    C = {t: pulp.LpVariable(f"C_{t}", lowBound=0) for t in T}
    O = {t: pulp.LpVariable(f"O_{t}", lowBound=0) for t in T}

    # 목적함수
    prob += pulp.lpSum(
        coeff_W*W[t] + coeff_O*O[t] + coeff_H*H[t] + coeff_L*L[t] +
        coeff_I*I[t] + coeff_S*S[t] + coeff_P*P[t] + coeff_C*C[t]
        for t in T
    )

    # 초기조건
    prob += W[0] == W0,  "W_init"
    prob += I[0] == I0,  "I_init"
    prob += S[0] == 0,   "S_init"

    # 제약조건
    for t in T:
        prob += W[t] == W[t-1] + H[t] - L[t],              f"labor_{t}"
        prob += P[t] <= cap_per_worker*W[t] + O[t]/std_time, f"capacity_{t}"
        prob += I[t] == I[t-1] + P[t] + C[t] - demand[t-1] - S[t-1] + S[t], f"inv_{t}"
        prob += O[t] <= max_ot_per_w*W[t],                  f"overtime_{t}"

    prob += I[TH] >= I_final_min, "I_final"
    prob += S[TH] == 0,           "S_final"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        return None

    res = {
        "W":  [pulp.value(W[t]) or 0 for t in T],
        "H":  [pulp.value(H[t]) or 0 for t in T],
        "L":  [pulp.value(L[t]) or 0 for t in T],
        "P":  [pulp.value(P[t]) or 0 for t in T],
        "I":  [pulp.value(I[t]) or 0 for t in T],
        "S":  [pulp.value(S[t]) or 0 for t in T],
        "C":  [pulp.value(C[t]) or 0 for t in T],
        "O":  [pulp.value(O[t]) or 0 for t in T],
        "total_cost": pulp.value(prob.objective),
        "status": pulp.LpStatus[status],
    }

    # 비용 분해
    res["cost_W"] = sum(coeff_W * res["W"][i] for i in range(TH))
    res["cost_O"] = sum(coeff_O * res["O"][i] for i in range(TH))
    res["cost_H"] = sum(coeff_H * res["H"][i] for i in range(TH))
    res["cost_L"] = sum(coeff_L * res["L"][i] for i in range(TH))
    res["cost_I"] = sum(coeff_I * res["I"][i] for i in range(TH))
    res["cost_S"] = sum(coeff_S * res["S"][i] for i in range(TH))
    res["cost_P"] = sum(coeff_P * res["P"][i] for i in range(TH))
    res["cost_C"] = sum(coeff_C * res["C"][i] for i in range(TH))

    revenue = sum(demand) * selling_p
    res["revenue"] = revenue
    res["profit"]  = revenue - res["total_cost"]
    res["svc_level"] = 1 - sum(res["S"]) / (sum(demand) + 1e-9)

    # ── 민감도 분석 ──
    # PuLP CBC는 dual 값을 직접 제공하지 않으므로
    # 수치적 방법으로 Shadow Price 근사 계산
    shadow = {}
    base_cost = res["total_cost"]

    # 수요 변화에 대한 Shadow Price (각 월 수요 +1 시 비용 변화)
    demand_shadow = []
    for t_idx in range(TH):
        d_perturbed = demand[:]
        d_perturbed[t_idx] += 1
        res2 = solve_app_simple(d_perturbed, coeff_W, coeff_O, coeff_H, coeff_L,
                                 coeff_I, coeff_S, coeff_P, coeff_C,
                                 cap_per_worker, max_ot_per_w, W0, I0, I_final_min)
        if res2:
            demand_shadow.append(res2["total_cost"] - base_cost)
        else:
            demand_shadow.append(None)

    # 초과시간 한도 변화에 대한 Shadow Price
    ot_shadow = []
    for t_idx in range(TH):
        # max_ot_per_w + 1 → 전체에 적용 (단순화)
        res3 = solve_app_simple(demand, coeff_W, coeff_O, coeff_H, coeff_L,
                                 coeff_I, coeff_S, coeff_P, coeff_C,
                                 cap_per_worker, max_ot_per_w + 1, W0, I0, I_final_min)
        if res3:
            ot_shadow.append(res3["total_cost"] - base_cost)
        else:
            ot_shadow.append(None)
        break  # 전체 기간에 동일하게 적용

    res["demand_shadow"] = demand_shadow
    res["ot_shadow_total"] = ot_shadow[0] if ot_shadow else None

    # 재고 제약 Shadow Price
    res3 = solve_app_simple(demand, coeff_W, coeff_O, coeff_H, coeff_L,
                             coeff_I, coeff_S, coeff_P, coeff_C,
                             cap_per_worker, max_ot_per_w, W0, I0, I_final_min + 1)
    res["inv_shadow"] = (res3["total_cost"] - base_cost) if res3 else None

    return res


def solve_app_simple(demand, coeff_W, coeff_O, coeff_H, coeff_L,
                     coeff_I, coeff_S, coeff_P, coeff_C,
                     cap_per_worker, max_ot_per_w, W0, I0, I_final_min):
    """Shadow Price 계산용 단순 버전 (결과값만 반환)"""
    TH = len(demand)
    T  = range(1, TH + 1)
    prob = pulp.LpProblem("APP_s", pulp.LpMinimize)
    W = {t: pulp.LpVariable(f"Ws_{t}", lowBound=0) for t in range(TH + 1)}
    H = {t: pulp.LpVariable(f"Hs_{t}", lowBound=0) for t in T}
    L = {t: pulp.LpVariable(f"Ls_{t}", lowBound=0) for t in T}
    P = {t: pulp.LpVariable(f"Ps_{t}", lowBound=0) for t in T}
    I = {t: pulp.LpVariable(f"Is_{t}", lowBound=0) for t in range(TH + 1)}
    S = {t: pulp.LpVariable(f"Ss_{t}", lowBound=0) for t in range(TH + 1)}
    C = {t: pulp.LpVariable(f"Cs_{t}", lowBound=0) for t in T}
    O = {t: pulp.LpVariable(f"Os_{t}", lowBound=0) for t in T}
    prob += pulp.lpSum(
        coeff_W*W[t] + coeff_O*O[t] + coeff_H*H[t] + coeff_L*L[t] +
        coeff_I*I[t] + coeff_S*S[t] + coeff_P*P[t] + coeff_C*C[t]
        for t in T
    )
    prob += W[0] == W0
    prob += I[0] == I0
    prob += S[0] == 0
    for t in T:
        prob += W[t] == W[t-1] + H[t] - L[t]
        prob += P[t] <= cap_per_worker*W[t] + O[t]/4
        prob += I[t] == I[t-1] + P[t] + C[t] - demand[t-1] - S[t-1] + S[t]
        prob += O[t] <= max_ot_per_w*W[t]
    prob += I[TH] >= I_final_min
    prob += S[TH] == 0
    st = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[st] != "Optimal":
        return None
    return {"total_cost": pulp.value(prob.objective)}


# # ──────────────────────────────────────────────
# # 전략 판별
# # ──────────────────────────────────────────────
# def classify_strategy(W_list, demand):
#     w_cv = np.std(W_list) / (np.mean(W_list) + 1e-9)
#     d_cv = np.std(demand)  / (np.mean(demand)  + 1e-9)
#     r    = w_cv / (d_cv + 1e-9)
#     if r < 0.25:
#         return ("수준 전략 (Level Strategy)", "#1565C0",
#                 "인력을 일정하게 유지 → 재고로 수요 변동 흡수. 재고비↑, 고용·해고비↓")
#     elif r > 0.65:
#         return ("수요 추종 전략 (Chase Strategy)", "#b71c1c",
#                 "수요에 맞춰 인력 조정 → 재고 최소화. 재고비↓, 고용·해고비↑")
#     else:
#         return ("혼합 전략 (Mixed Strategy)", "#e65100",
#                 "인력 조정 + 재고 완충의 균형 전략")


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────
st.title("🌿 총괄생산계획 최적화 APP")
st.caption("홍익대학교 산업데이터공학과 C421086 김지은")

with st.spinner("⚙️ 최적화 + 민감도 분석 계산 중..."):
    res = solve_app(demand, coeff_W, coeff_O, coeff_H, coeff_L,
                    coeff_I, coeff_S, coeff_P, coeff_C,
                    cap_per_worker, max_ot_per_w, W0, I0, I_final_min)

if res is None:
    st.error("❌ 최적해를 찾지 못했습니다. 파라미터를 조정해 주세요.")
    st.stop()

TH = len(demand)

# # ── 전략 배지 ─────────────────────────────────
# s_name, s_color, s_desc = classify_strategy(res["W"], demand)
# st.markdown(f"""
# <div class="strategy-box" style="background:{s_color}15; border-left:5px solid {s_color};">
#   <span class="badge" style="background:{s_color}; color:white;">📊 {s_name}</span>
#   <span style="color:#333; font-size:13px;">{s_desc}</span>
# </div>
# """, unsafe_allow_html=True)

# ── KPI 카드 ──────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
def kpi(col, label, value, color, sub=""):
    col.markdown(f"""
    <div class="kpi-box">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{color};">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, "총 비용",    f"₩{res['total_cost']:,.0f}천원", "#e53935", "최소화 목적함수")
kpi(k2, "예상 이익",  f"₩{res['profit']:,.0f}천원",     "#43a047", f"매출 ₩{res['revenue']:,.0f}")
kpi(k3, "서비스 수준",f"{res['svc_level']*100:.1f}%",   "#1e88e5",
    "✅ 부족 없음" if res['svc_level'] >= 1 else f"부족 {sum(res['S']):,.0f}개")
kpi(k4, "총 외주량",  f"{sum(res['C']):,.0f}개",        "#fb8c00", f"총생산 {sum(res['P'])+sum(res['C']):,.0f}개")
kpi(k5, "수요 Shadow Price",
    f"₩{np.mean([v for v in res['demand_shadow'] if v is not None]):,.1f}천원",
    "#7b1fa2", "수요 1개↑ 시 평균 비용 변화")

st.markdown("<br>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────
# 탭 구성
# ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 생산계획 대시보드",
    "🔍 민감도 분석",
    "📐 LP 시각화",
    "📋 상세 데이터"
])

# ══════════════════════════════════════════════
# TAB 1: 생산계획 대시보드
# ══════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📊 월별 생산량 구성 — 정규 / 초과 / 외주</div>',
                    unsafe_allow_html=True)
        reg_prod = [cap_per_worker * res["W"][i] for i in range(TH)]
        ot_prod  = [res["O"][i] / std_time for i in range(TH)]

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(name="정규생산", x=MONTHS_6, y=reg_prod,
                              marker_color="#388e3c",
                              text=[f"{v:.0f}" for v in reg_prod], textposition="inside",
                              textfont_color="white"))
        fig1.add_trace(go.Bar(name="초과근무생산", x=MONTHS_6, y=ot_prod,
                              marker_color="#f9a825",
                              text=[f"{v:.0f}" for v in ot_prod], textposition="inside"))
        fig1.add_trace(go.Bar(name="외주(C_t)", x=MONTHS_6, y=res["C"],
                              marker_color="#e53935",
                              text=[f"{v:.0f}" for v in res["C"]], textposition="inside",
                              textfont_color="white"))
        fig1.add_trace(go.Scatter(name="수요(D_t)", x=MONTHS_6, y=demand,
                                  mode="lines+markers+text",
                                  line=dict(color="#1565c0", width=2.5, dash="dash"),
                                  marker=dict(size=9, symbol="diamond"),
                                  text=[f"{v}" for v in demand],
                                  textposition="top center",
                                  textfont=dict(color="#1565c0", size=11)))
        fig1.update_layout(barmode="stack", height=330, margin=dict(t=10,b=50,l=10,r=10),
                           legend=dict(orientation="h", y=-0.2),
                           plot_bgcolor="white", paper_bgcolor="white",
                           yaxis_title="수량 (개)")
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📦 재고·부족 흐름</div>',
                    unsafe_allow_html=True)
        inv_full   = [I0] + res["I"]
        months_ext = ["초기"] + MONTHS_6
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(name="기말재고(I_t)", x=months_ext, y=inv_full,
                                  fill="tozeroy", fillcolor="rgba(56,142,60,0.12)",
                                  line=dict(color="#388e3c", width=2.5),
                                  mode="lines+markers",
                                  text=[f"{v:.0f}" for v in inv_full],
                                  textposition="top center"))
        if any(s > 0.1 for s in res["S"]):
            fig2.add_trace(go.Bar(name="부족분(S_t)", x=MONTHS_6, y=res["S"],
                                  marker_color="rgba(229,57,53,0.75)"))
        fig2.add_hline(y=I_final_min, line_dash="dot", line_color="#fb8c00",
                       annotation_text=f"기말최소 {I_final_min}개")
        fig2.update_layout(height=330, margin=dict(t=10,b=50,l=10,r=10),
                           legend=dict(orientation="h", y=-0.2),
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col_c, col_d = st.columns([2, 3])

    with col_c:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🧑‍🏭 인력 변동 (W_t / H_t / L_t)</div>',
                    unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(name="인력(W_t)", x=MONTHS_6, y=res["W"],
                                  fill="tozeroy", fillcolor="rgba(30,136,229,0.10)",
                                  line=dict(color="#1e88e5", width=3),
                                  mode="lines+markers+text",
                                  text=[f"{v:.1f}" for v in res["W"]],
                                  textposition="top center",
                                  textfont=dict(size=10, color="#1565c0"),
                                  marker=dict(size=8)))
        fig3.add_trace(go.Bar(name="고용(H_t)", x=MONTHS_6, y=res["H"],
                              marker_color="rgba(67,160,71,0.8)"))
        fig3.add_trace(go.Bar(name="해고(L_t)", x=MONTHS_6, y=[-v for v in res["L"]],
                              marker_color="rgba(229,57,53,0.8)"))
        fig3.add_hline(y=W0, line_dash="dot", line_color="#aaa",
                       annotation_text=f"초기 W₀={W0}")
        fig3.update_layout(barmode="relative", height=310,
                           margin=dict(t=10,b=50,l=10,r=10),
                           legend=dict(orientation="h", y=-0.22),
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">💸 비용 구성 분석</div>',
                    unsafe_allow_html=True)
        cost_items = {
            "정규노동비": res["cost_W"], "초과근무비": res["cost_O"],
            "고용비":     res["cost_H"], "해고비":     res["cost_L"],
            "재고유지비": res["cost_I"], "부족비":     res["cost_S"],
            "재료비":     res["cost_P"], "하청비":     res["cost_C"],
        }
        colors = ["#388e3c","#f9a825","#66bb6a","#e53935",
                  "#42a5f5","#ff7043","#ab47bc","#ff9800"]
        labels, values, clrs = zip(*[
            (l, v, c) for (l, v), c in zip(cost_items.items(), colors) if v > 0.01
        ])
        fig4 = go.Figure(go.Pie(
            labels=labels, values=values, marker_colors=clrs,
            hole=0.46, textinfo="label+percent", textfont=dict(size=11),
            hovertemplate="%{label}<br><b>₩%{value:,.0f}천원</b><br>%{percent}<extra></extra>",
            sort=True, direction="clockwise"
        ))
        fig4.add_annotation(
            text=f"총비용<br><b>₩{res['total_cost']:,.0f}</b><br>천원",
            x=0.5, y=0.5, showarrow=False, font=dict(size=11, color="#333")
        )
        fig4.update_layout(height=310, margin=dict(t=10,b=20,l=10,r=10),
                           legend=dict(orientation="v", x=1.0, font=dict(size=10)),
                           paper_bgcolor="white")
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2: 민감도 분석
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔍 민감도 분석 (Sensitivity Analysis)</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="tab-note">'
                '각 파라미터가 1단위 변화할 때 최적해 및 목적함수 값이 얼마나 달라지는지를 계산</div>',
                unsafe_allow_html=True)

    # ── 2-1: 월별 수요 Shadow Price ──────────
    st.markdown("#### 📌 수요 잠재가격 (Demand Shadow Price)")
    st.caption("각 월의 수요가 1개 증가할 때 목적함수 값의 변화량")

    col_sa, col_sb = st.columns([2, 3])
    with col_sa:
        shadow_df = pd.DataFrame({
            "월": MONTHS_6,
            "수요(D_t)": demand,
            "Shadow Price(천원/개)": [
                round(v, 3) if v is not None else "N/A"
                for v in res["demand_shadow"]
            ],
            "해석": [
                f"수요 1↑ → 비용 {v:+.2f}천원" if v is not None else "-"
                for v in res["demand_shadow"]
            ]
        })
        st.dataframe(shadow_df.set_index("월"), use_container_width=True)

    with col_sb:
        valid_shadow = [(m, v) for m, v in zip(MONTHS_6, res["demand_shadow"]) if v is not None]
        if valid_shadow:
            months_v, vals_v = zip(*valid_shadow)
            bar_colors = ["#e53935" if v > 0 else "#43a047" for v in vals_v]
            fig_s1 = go.Figure(go.Bar(
                x=list(months_v), y=list(vals_v),
                marker_color=bar_colors,
                text=[f"{v:+.2f}" for v in vals_v],
                textposition="outside",
            ))
            fig_s1.update_layout(
                title="수요 Shadow Price — 월별 비교",
                height=300, margin=dict(t=40,b=40,l=10,r=10),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis_title="비용 변화 (천원/개)",
                yaxis=dict(zeroline=True, zerolinecolor="#aaa", zerolinewidth=2)
            )
            st.plotly_chart(fig_s1, use_container_width=True)

    # ── 2-2: 자원 제약 Shadow Price ──────────
    st.markdown("#### 📌 자원 제약 잠재가격")

    # 자원 제약 카드 2개 나란히
    rc1, rc2 = st.columns(2)
    ot_sp  = res.get("ot_shadow_total")
    inv_sp = res.get("inv_shadow")
    with rc1:
        st.markdown(f"""
        <div class="shadow-card">
          <b>⏰ 초과시간 한도 (Hr/인·월)</b><br>
          현재 한도: <b>{max_ot_per_w}시간</b><br>
          한도 1시간 증가 시 총비용 변화:
          <b style="color:#1565c0;">{f'{ot_sp:+.2f}천원' if ot_sp is not None else 'N/A'}</b><br>
          <small>→ {'음수이면 초과시간 한도 확대가 비용 절감에 기여' if ot_sp and ot_sp < 0 else '초과시간 확대가 비용절감에 기여하지 않음'}</small>
        </div>
        """, unsafe_allow_html=True)
    with rc2:
        st.markdown(f"""
        <div class="shadow-card">
          <b>📦 기말 최소재고 제약</b><br>
          현재 기말재고 목표: <b>{I_final_min}개</b><br>
          목표 1개 증가 시 총비용 변화:
          <b style="color:#e53935;">{f'{inv_sp:+.2f}천원' if inv_sp is not None else 'N/A'}</b><br>
          <small>→ 기말재고 요구량이 엄격할수록 비용 증가</small>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    # 비용계수 상대적 크기 비교 — 전체 너비
    st.markdown("#### 📌 비용계수 상대적 크기 비교")
    st.caption("각 비용 계수의 상대적 크기. 계수가 클수록 해당 변수가 목적함수에 미치는 단위당 영향이 큼")

    coeff_names = ["정규노동비", "초과임금", "고용비", "해고비",
                   "재고비", "부족비", "재료비", "하청비"]
    coeff_vals_list = [coeff_W, coeff_O, coeff_H, coeff_L, coeff_I, coeff_S, coeff_P, coeff_C]
    total_c = sum(coeff_vals_list)
    pcts    = [v / total_c * 100 for v in coeff_vals_list]

    fig_s2 = go.Figure(go.Bar(
        x=pcts, y=coeff_names, orientation='h',
        marker_color=["#388e3c","#f9a825","#66bb6a","#e53935",
                      "#42a5f5","#ff7043","#ab47bc","#ff9800"],
        text=[f"{v:.1f}%" for v in pcts], textposition="outside"
    ))
    fig_s2.update_layout(
        height=320, margin=dict(t=20, b=20, l=10, r=80),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="비중 (%)",
        xaxis=dict(range=[0, max(pcts) * 1.25])
    )
    st.plotly_chart(fig_s2, use_container_width=True)




# ══════════════════════════════════════════════
# TAB 3: LP 시각화
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📐 LP 제약조건 시각화</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="tab-note">'
                '특정 기간을 선택하여 인력(W_t)과 생산량(P_t)의 관계를 시각화</div>',
                unsafe_allow_html=True)

    sel_month = st.selectbox("분석 기간 선택", MONTHS_6, index=0)
    t_idx = MONTHS_6.index(sel_month)
    D_t   = demand[t_idx]
    W_opt = res["W"][t_idx]
    P_opt = res["P"][t_idx]
    O_opt = res["O"][t_idx]

    st.markdown(f"**{sel_month} 최적해**: W={W_opt:.1f}명, P={P_opt:.1f}개, O={O_opt:.1f}Hr")

    # LP 시각화: X축=인력(W), Y축=생산량(P)
    W_max = max(W_opt * 2, 120)
    P_max = max(P_opt * 2, D_t * 1.5)
    W_range = np.linspace(0, W_max, 300)

    fig_lp = go.Figure()

    # 제약 1: P ≤ 40W + O/4  (O=10W 최대 잔업 대입 시 P ≤ 40W + 10W/4 = 42.5W)
    P_cap_normal = cap_per_worker * W_range
    P_cap_max_ot = (cap_per_worker + max_ot_per_w / std_time) * W_range

    fig_lp.add_trace(go.Scatter(
        x=W_range, y=P_cap_normal, name=f"정규생산 최대 (P≤{cap_per_worker}W)",
        line=dict(color="#388e3c", width=2.5),
        mode="lines"
    ))
    fig_lp.add_trace(go.Scatter(
        x=W_range, y=P_cap_max_ot,
        name=f"정규+최대잔업 (P≤{cap_per_worker+max_ot_per_w/std_time:.1f}W)",
        line=dict(color="#f9a825", width=2.5, dash="dash"),
        mode="lines"
    ))

    # 제약 2: 수요 충족선 P = D_t (외주 제외 시)
    fig_lp.add_hline(y=D_t, line_dash="dot", line_color="#1565c0",
                     line_width=2,
                     annotation_text=f"수요 D_{t_idx+1}={D_t}개",
                     annotation_position="right")

    # 실행가능영역 (정규생산 기준) - fill between
    W_feasible = W_range[P_cap_normal <= P_max]
    P_feasible = P_cap_normal[:len(W_feasible)]
    fig_lp.add_trace(go.Scatter(
        x=np.append(W_feasible, W_feasible[::-1]),
        y=np.append(np.minimum(P_feasible, P_max), np.zeros(len(W_feasible))),
        fill="toself", fillcolor="rgba(56,142,60,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="정규 실행가능영역", showlegend=True
    ))

    # 등고선 (비용 = coeff_W·W + coeff_P·P = const)
    for z_val in np.linspace(
        coeff_W * W_opt * 0.5 + coeff_P * P_opt * 0.5,
        coeff_W * W_opt * 1.5 + coeff_P * P_opt * 1.5, 5
    ):
        # coeff_W·W + coeff_P·P = z → P = (z - coeff_W·W)/coeff_P
        P_iso = (z_val - coeff_W * W_range) / coeff_P
        mask  = (P_iso >= 0) & (P_iso <= P_max)
        fig_lp.add_trace(go.Scatter(
            x=W_range[mask], y=P_iso[mask],
            line=dict(color="rgba(156,39,176,0.4)", dash="dot", width=1),
            mode="lines", showlegend=False,
            hovertemplate=f"비용등고선 Z={z_val:.0f}"
        ))

    # 최적해 표시
    fig_lp.add_trace(go.Scatter(
        x=[W_opt], y=[P_opt],
        mode="markers+text",
        marker=dict(color="#e53935", size=14, symbol="star"),
        text=[f"최적해<br>W={W_opt:.1f}, P={P_opt:.1f}"],
        textposition="top right",
        textfont=dict(size=12, color="#b71c1c"),
        name="최적해"
    ))

    fig_lp.update_layout(
        title=f"{sel_month} — 인력(W) vs 생산량(P) 제약조건 시각화",
        height=480,
        xaxis_title="인력 W_t (명)",
        yaxis_title="생산량 P_t (개)",
        xaxis=dict(range=[0, W_max]),
        yaxis=dict(range=[0, P_max]),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=50, b=80, l=10, r=10)
    )
    st.plotly_chart(fig_lp, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="formula-box">
    📐 {sel_month} 제약 요약<br>
    - 생산능력:  P_{t_idx+1} ≤ {cap_per_worker}·W_{t_idx+1} + O_{t_idx+1}/4<br>
    - 초과근무:  O_{t_idx+1} ≤ {max_ot_per_w}·W_{t_idx+1}<br>
    - 수요:      D_{t_idx+1} = {D_t}개 (외주 포함 시 P+C ≥ D)<br>
    - 최적해:    W={W_opt:.2f}명, P={P_opt:.2f}개, O={O_opt:.2f}Hr
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # ── LP 제약조건 설명 ──
    st.markdown("#### 📋 제약조건 및 실행가능해 분석")
    lp1, lp2 = st.columns(2)

    with lp1:
        # 현재 월 제약 충족 여부 확인
        cap_used_normal = P_opt / (cap_per_worker * W_opt + 1e-9) * 100 if W_opt > 0 else 0
        cap_used_total  = P_opt / ((cap_per_worker + max_ot_per_w/std_time) * W_opt + 1e-9) * 100 if W_opt > 0 else 0
        ot_used_pct     = O_opt / (max_ot_per_w * W_opt + 1e-9) * 100 if W_opt > 0 else 0

        st.markdown(f"""
        <div class="shadow-card">
          <b>⚙️ {sel_month} 제약조건 충족 현황</b><br><br>
          <b>생산능력 (정규 기준)</b>: {P_opt:.0f} / {cap_per_worker * W_opt:.0f}개
          → <b style="color:{'#e53935' if cap_used_normal > 95 else '#388e3c'};">
          {cap_used_normal:.1f}% 사용</b><br>
          <b>생산능력 (정규+잔업)</b>: {P_opt:.0f} / {(cap_per_worker + max_ot_per_w/std_time)*W_opt:.0f}개
          → <b>{cap_used_total:.1f}% 사용</b><br>
          <b>초과시간 사용률</b>: {O_opt:.1f} / {max_ot_per_w * W_opt:.1f}Hr
          → <b>{ot_used_pct:.1f}% 사용</b><br>
          <b>외주 투입량</b>: {res["C"][t_idx]:.0f}개<br><br>
          <small>→ 생산능력 사용률이 100%면 바인딩 제약 (여유 없음)</small>
        </div>
        """, unsafe_allow_html=True)

    with lp2:
        # 실행가능해 판단
        supply_total = P_opt + res["C"][t_idx] + (res["I"][t_idx-1] if t_idx > 0 else I0)
        feasible_check = supply_total >= D_t
        st.markdown(f"""
        <div class="shadow-card">
          <b>✅ {sel_month} 실행가능해 판단</b><br><br>
          <b>수요(D_t)</b>: {D_t:,}개<br>
          <b>정규생산(P_t)</b>: {P_opt:.0f}개<br>
          <b>외주(C_t)</b>: {res["C"][t_idx]:.0f}개<br>
          <b>초과시간(O_t)</b>: {O_opt:.1f}Hr → 추가생산 {O_opt/std_time:.0f}개<br>
          <b>부족분(S_t)</b>: {res["S"][t_idx]:.0f}개<br><br>
          <b style="color:{'#388e3c' if res['S'][t_idx] < 0.1 else '#e53935'};">
          → {'✅ 수요 완전 충족 (실행가능해)' if res["S"][t_idx] < 0.1 else f'⚠️ 부족 {res["S"][t_idx]:.0f}개 발생'}</b>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4: 상세 데이터
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📋 월별 최적화 결과 상세</div>', unsafe_allow_html=True)

    df = pd.DataFrame({
        "월":           MONTHS_6,
        "수요 D_t":     demand,
        "인력 W_t":     [round(v, 2) for v in res["W"]],
        "고용 H_t":     [round(v, 2) for v in res["H"]],
        "해고 L_t":     [round(v, 2) for v in res["L"]],
        "생산 P_t":     [round(v, 2) for v in res["P"]],
        "재고 I_t":     [round(v, 2) for v in res["I"]],
        "부족 S_t":     [round(v, 2) for v in res["S"]],
        "외주 C_t":     [round(v, 2) for v in res["C"]],
        "초과시간 O_t": [round(v, 2) for v in res["O"]],
        "수요 Shadow":  [round(v, 3) if v is not None else None for v in res["demand_shadow"]],
    })
    st.dataframe(df.set_index("월"), use_container_width=True)

    st.info(f"📌 기본값 최적비용: **422,275천원** | 현재: **{res['total_cost']:,.0f}천원**"
            + ("  ✅ 기본값과 일치" if abs(res['total_cost'] - 422275) < 1 else "  (파라미터 변경됨)"))

    # 월별 비용 상세
    st.markdown("##### 월별 비용 상세 분해")
    cost_df = pd.DataFrame({
        "월":       MONTHS_6,
        "정규노동비": [coeff_W*res["W"][i] for i in range(TH)],
        "초과근무비": [coeff_O*res["O"][i] for i in range(TH)],
        "고용비":    [coeff_H*res["H"][i] for i in range(TH)],
        "해고비":    [coeff_L*res["L"][i] for i in range(TH)],
        "재고유지비": [coeff_I*res["I"][i] for i in range(TH)],
        "재료비":    [coeff_P*res["P"][i] for i in range(TH)],
        "하청비":    [coeff_C*res["C"][i] for i in range(TH)],
        "부족비":    [coeff_S*res["S"][i] for i in range(TH)],
    })
    cost_df["합계"] = cost_df.iloc[:, 1:].sum(axis=1)
    st.dataframe(cost_df.set_index("월").style.format("{:.1f}"), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ CSV 다운로드", csv, "APP_v2_결과.csv", "text/csv")

    # 목적함수 수식
    with st.expander("📐 목적함수 수식"):
        st.markdown(f"""
        <div class="formula-box">
        Z = Σ[ {coeff_W}·W_t + {coeff_O}·O_t + {coeff_H}·H_t + {coeff_L}·L_t
             + {coeff_I}·I_t + {coeff_S}·S_t + {coeff_P}·P_t + {coeff_C}·C_t ]<br><br>
        {coeff_W} = 정규임금({c_regular}) × 작업시간({work_hours}) × 작업일수({work_days})
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("🌿 홍익대학교 산업데이터공학과 C421086 김지은")