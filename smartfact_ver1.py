"""
원예장비 제조업체 총괄생산계획(APP) 최적화 웹앱
Aggregate Production Planning Optimization with PuLP + Streamlit
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import pulp
import plotly.graph_objects as go
import plotly.express as px

# 1. 페이지 설정
st.set_page_config(page_title="스마트제조 APP 최적화 시스템", layout="wide")
st.title("🚜 원예장비 제조업체 총괄생산계획(APP) 의사결정 지원 시스템")

# 2. 사이드바: 파라미터 및 수요 입력
with st.sidebar:
    st.header("📋 입력 파라미터 설정")
    
    # 강의록 기본 데이터 초기값 설정
    with st.expander("📅 수요 예측 수정 (1~6월)", expanded=True):
        d1 = st.number_input("1월 수요", value=1600)
        d2 = st.number_input("2월 수요", value=3000)
        d3 = st.number_input("3월 수요", value=3200)
        d4 = st.number_input("4월 수요", value=3800)
        d5 = st.number_input("5월 수요", value=2200)
        d6 = st.number_input("6월 수요", value=2200)
        demand = [d1, d2, d3, d4, d5, d6]

    with st.expander("💰 단위 비용 설정"):
        c_hire = st.slider("고용 비용 (천원/인)", 100, 1000, 300)
        c_fire = st.slider("해고 비용 (천원/인)", 100, 1000, 500)
        c_hold = st.number_input("재고유지비 (천원/개/월)", value=2)
        c_back = st.number_input("부재고비용 (천원/개/월)", value=5)
        c_sub = st.number_input("하청비용 (천원/개)", value=30)
        c_reg = st.number_input("정규생산비용 (천원/개)", value=10)
        c_ot = st.number_input("잔업 비용 (천원/시간)", value=6)

    with st.expander("⚙️ 운영 제약 조건"):
        ot_limit = st.slider("인당 최대 잔업 시간 (h)", 0, 40, 10)
        initial_w = st.number_input("초기 작업자 수", value=80)
        initial_i = st.number_input("초기 재고량", value=1000)

# 3. 최적화 엔진 함수
def solve_app(demand_list, hire_c, fire_c, ot_lim):
    T = len(demand_list)
    periods = range(T)
    
    # 문제 정의
    prob = pulp.LpProblem("APP_Optimization", pulp.LpMinimize)

    # 결정변수 (0 이상의 실수)
    W = [pulp.LpVariable(f'W_{t}', lowBound=0) for t in range(T + 1)] # 인력
    H = [pulp.LpVariable(f'H_{t}', lowBound=0) for t in range(T + 1)] # 고용
    L = [pulp.LpVariable(f'L_{t}', lowBound=0) for t in range(T + 1)] # 해고
    P = [pulp.LpVariable(f'P_{t}', lowBound=0) for t in range(T + 1)] # 생산량
    I = [pulp.LpVariable(f'I_{t}', lowBound=0) for t in range(T + 1)] # 재고
    S = [pulp.LpVariable(f'S_{t}', lowBound=0) for t in range(T + 1)] # 부족분
    C = [pulp.LpVariable(f'C_{t}', lowBound=0) for t in range(T + 1)] # 외주
    O = [pulp.LpVariable(f'O_{t}', lowBound=0) for t in range(T + 1)] # 잔업시간

    # 목적함수: 총 비용 최소화
    prob += pulp.lpSum([
        640 * W[t+1] + c_ot * O[t+1] + hire_c * H[t+1] + fire_c * L[t+1] +
        c_hold * I[t+1] + c_back * S[t+1] + c_reg * P[t+1] + c_sub * C[t+1]
        for t in periods
    ])

    # 초기 조건 제약
    prob += W[0] == initial_w
    prob += I[0] == initial_i
    prob += S[0] == 0

    # 월별 제약 조건
    for t in periods:
        prob += W[t+1] == W[t] + H[t+1] - L[t+1] # 인력 균형
        prob += P[t+1] <= 40 * W[t+1] + 0.25 * O[t+1] # 생산 능력 (시간당 4개 가정 시 0.25h/개)
        prob += I[t+1] - S[t+1] == I[t] - S[t] + P[t+1] + C[t+1] - demand_list[t] # 재고 균형
        prob += O[t+1] <= ot_lim * W[t+1] # 잔업 제한

    # 기말 조건
    prob += I[T] >= 500
    prob += S[T] == 0

    # 최적화 실행
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        res = {
            'TotalCost': pulp.value(prob.objective),
            'Workers': [pulp.value(W[t]) for t in range(1, T+1)],
            'Production': [pulp.value(P[t]) for t in range(1, T+1)],
            'Inventory': [pulp.value(I[t]) for t in range(1, T+1)],
            'Subcontract': [pulp.value(C[t]) for t in range(1, T+1)],
            'Overtime': [pulp.value(O[t]) for t in range(1, T+1)]
        }
        return res
    return None

# 4. 결과 도출 및 시각화
res = solve_app(demand, c_hire, c_fire, ot_limit)

if res:
    # 핵심 지표 표시
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💰 총 비용", f"{res['TotalCost']:,.0f} 천원")
    m2.metric("👥 최대 인력", f"{max(res['Workers']):.0f} 명")
    m3.metric("📦 기말 재고", f"{res['Inventory'][-1]:.0f} 개")
    m4.metric("🏢 외주 비중", f"{(sum(res['Subcontract'])/sum(demand)*100):.1f} %")

    # 대시보드 레이아웃
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 생산 및 수요 밸런스")
        df_plot = pd.DataFrame({
            '월': [f'{i+1}월' for i in range(6)],
            '수요': demand,
            '내부생산': res['Production'],
            '외주(하청)': res['Subcontract']
        })
        fig = px.bar(df_plot, x='월', y=['내부생산', '외주(하청)'], title="월별 생산 계획 (누적)")
        fig.add_scatter(x=df_plot['월'], y=df_plot['수요'], name='수요 예측', mode='lines+markers', line=dict(color='red'))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🧪 민감도 분석: 해고 비용의 영향")
        f_costs = np.linspace(100, 1000, 10)
        s_results = []
        for fc in f_costs:
            s_res = solve_app(demand, c_hire, fc, ot_limit)
            s_results.append(s_res['TotalCost'] if s_res else None)
        
        fig_sens = px.line(x=f_costs, y=s_results, markers=True, 
                           labels={'x': '해고 비용 단가', 'y': '총 최적 비용'},
                           title="해고 비용 변화에 따른 총 운영 비용 추이")
        st.plotly_chart(fig_sens, use_container_width=True)

    # 상세 데이터 테이블
    with st.expander("📄 최적 생산 계획 데이터 상세 보기"):
        st.table(pd.DataFrame(res, index=[f'{i+1}월' for i in range(6)]))

else:
    st.error("❌ 최적해를 찾을 수 없습니다. 제약 조건을 확인해 주세요.")