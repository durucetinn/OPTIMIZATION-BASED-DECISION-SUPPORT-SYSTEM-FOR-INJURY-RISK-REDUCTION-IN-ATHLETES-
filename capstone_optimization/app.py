import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_loader import (
    generate_raw_criteria,
    compute_owa_score,
    get_aggregated_ahp_weights,
    CRITERIA,
    CRITERIA_TR,
    EXPERT_AHP_WEIGHTS
)
from optimization_models import (
    solve_risk_prioritization,
    solve_threshold_classification,
    solve_asymmetric_cost_optimization,
    run_sensitivity_analysis_costs
)

CRITERIA_EN = {
    "Ankle_Foot_Mechanics": "Ankle & Foot Mechanics",
    "Knee_Mechanics": "Knee Mechanics",
    "Hip_Mechanics": "Hip Mechanics",
    "Trunk_Control": "Trunk Control",
    "Stance_Lower_Extremity": "Stance & Lower Extremity",
    "Overall_Landing_Quality": "Overall Landing Quality / Motor Control"
}

# Mapping dictionaries
AHP_METHOD_MAP = {
    "Neutral / Average": {"Türkçe": "Nötr / Ortalama", "English": "Neutral / Average"},
    "Optimistic / Max": {"Türkçe": "İyimser / Maks", "English": "Optimistic / Max"},
    "Pessimistic / Min": {"Türkçe": "Kötümser / Min", "English": "Pessimistic / Min"},
    "CR-Based": {"Türkçe": "Tutarlılık Oranı (CR) Tabanlı", "English": "CR-Based"},
    "Confidence-Based": {"Türkçe": "Güven Derecesi Tabanlı", "English": "Confidence-Based"},
}

OWA_PROFILE_MAP = {
    "conservative": {
        "Türkçe": "Muhafazakar (En Yüksek Riske Büyük Ağırlık)",
        "English": "Conservative (Heavy weight on highest risk)"
    },
    "equal": {
        "Türkçe": "Eşit Ağırlıklı (Aritmetik Ortalama)",
        "English": "Equal Weighted (Arithmetic Mean)"
    },
    "custom": {
        "Türkçe": "Özel Ağırlıklar (Sürgülerle Ayarlayın)",
        "English": "Custom Weights (Adjust with Sliders)"
    }
}

RISK_CLASS_MAP = {
    "Düşük Risk": {"Türkçe": "Düşük Risk", "English": "Low Risk"},
    "Orta Risk": {"Türkçe": "Orta Risk", "English": "Medium Risk"},
    "Yüksek Risk": {"Türkçe": "Yüksek Risk", "English": "High Risk"}
}

# Set page configuration for a premium, wide dashboard look
st.set_page_config(
    page_title="Klinik Karar Destek & Optimizasyon / Clinical Decision Support & Optimization",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Language selector in sidebar as the very first element
lang = st.sidebar.radio("🌐 Dil / Language", ["Türkçe", "English"], horizontal=True)

# Select translation maps based on language
crit_map = CRITERIA_TR if lang == "Türkçe" else CRITERIA_EN

# Inject custom CSS for premium styling
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    .reportview-container {
        font-family: 'Inter', sans-serif;
    }
    h1 {
        color: #0f172a;
        font-weight: 850;
        margin-bottom: 2px !important;
    }
    .subtitle {
        color: #475569;
        font-size: 1.15rem;
        margin-bottom: 20px;
        font-weight: 400;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .category-low {
        color: #10b981;
        font-weight: bold;
    }
    .category-med {
        color: #f59e0b;
        font-weight: bold;
    }
    .category-high {
        color: #ef4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App Header
if lang == "Türkçe":
    st.write("<h1>🏥 Klinik Karar Destek ve Sakatlık Riski Optimizasyon Paneli</h1>", unsafe_allow_html=True)
    st.write("<div class='subtitle'>AHP, OWA ve Asimetrik Klinik Hata Maliyeti Karar Modelleri (Capstone Projesi)</div>", unsafe_allow_html=True)
else:
    st.write("<h1>🏥 Clinical Decision Support and Injury Risk Optimization Dashboard</h1>", unsafe_allow_html=True)
    st.write("<div class='subtitle'>AHP, OWA and Asymmetric Clinical Error Cost Decision Models (Capstone Project)</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# Sidebar Settings
# ----------------------------------------------------
st.sidebar.header("⚙️ Genel Parametreler" if lang == "Türkçe" else "⚙️ General Parameters")

n_athletes = st.sidebar.slider("Atlet Sayısı (N)" if lang == "Türkçe" else "Number of Athletes (N)", 10, 100, 30, step=5)
seed = st.sidebar.number_input("Veri Üretim Seed (Rastgelelik)" if lang == "Türkçe" else "Data Generation Seed (Randomness)", value=42, step=1)

# AHP Aggregation Choice
st.sidebar.subheader("⚖️ AHP Ağırlık Aggregasyonu" if lang == "Türkçe" else "⚖️ AHP Weight Aggregation")
ahp_options = [AHP_METHOD_MAP[k][lang] for k in AHP_METHOD_MAP.keys()]
ahp_choice_display = st.sidebar.selectbox("AHP Senaryosu Seçin" if lang == "Türkçe" else "Select AHP Scenario", ahp_options)
# Map back to English key for function call
ahp_method = [k for k, v in AHP_METHOD_MAP.items() if v[lang] == ahp_choice_display][0]

# OWA Profile Choice
st.sidebar.subheader("🌀 OWA Ağırlıkları (Sıralı Birleştirme)" if lang == "Türkçe" else "🌀 OWA Weights (Ordered Weighted Averaging)")
owa_options = [OWA_PROFILE_MAP[k][lang] for k in OWA_PROFILE_MAP.keys()]
owa_choice_display = st.sidebar.selectbox("OWA Senaryosu Seçin" if lang == "Türkçe" else "Select OWA Scenario", owa_options)

if owa_choice_display == OWA_PROFILE_MAP["conservative"][lang]:
    # 6 weights corresponding to sorted criteria
    owa_weights = np.array([0.35, 0.25, 0.18, 0.12, 0.07, 0.03])
elif owa_choice_display == OWA_PROFILE_MAP["equal"][lang]:
    owa_weights = np.array([1/6] * 6)
else:
    raw_owa = []
    if lang == "Türkçe":
        st.sidebar.markdown("<small>Ağırlıklar azalan düzende sıralı girilmelidir.</small>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<small>Weights must be sorted in descending order.</small>", unsafe_allow_html=True)
    for i in range(6):
        label = f"{i+1}. En Kötü Kriter Ağırlığı" if lang == "Türkçe" else f"{i+1}. {i+1} Worst Criterion Weight"
        w = st.sidebar.slider(label, 0.0, 1.0, float(0.30 - i * 0.05) if 0.30 - i * 0.05 > 0 else 0.01)
        raw_owa.append(w)
    owa_weights = np.array(raw_owa)
    owa_weights = owa_weights / np.sum(owa_weights)

# Model specific parameters
st.sidebar.header("🎯 Model Parametreleri" if lang == "Türkçe" else "🎯 Model Parameters")

st.sidebar.markdown("### Model 1: Risk Prioritization")
M_capacity = st.sidebar.slider("Maksimum İzlenebilecek Atlet Sayısı (M)" if lang == "Türkçe" else "Maximum Athletes to Monitor (M)", 1, n_athletes, min(10, n_athletes))

st.sidebar.markdown("### Model 2: Threshold-Based")
T_threshold = st.sidebar.slider("Risk Eşik Değeri (T)" if lang == "Türkçe" else "Risk Threshold Value (T)", 1.0, 10.0, 5.5, 0.5)

st.sidebar.markdown("### Model 3: Asymmetric Clinical Cost")
c_fn = st.sidebar.slider(
    "Yanlış Negatif Klinik Maliyeti (C_FN)" if lang == "Türkçe" else "False Negative Clinical Cost (C_FN)", 
    1.0, 15.0, 9.0, 0.5, 
    help="Yüksek riskli atleti kaçırmanın klinik maliyeti (Varsayılan = 9)" if lang == "Türkçe" else "Clinical cost of missing a high-risk athlete (Default = 9)"
)
c_fp = st.sidebar.slider(
    "Yanlış Pozitif Klinik Maliyeti (C_FP)" if lang == "Türkçe" else "False Positive Clinical Cost (C_FP)", 
    1.0, 15.0, 3.0, 0.5, 
    help="Sağlıklı atlete gereksiz müdahalenin klinik maliyeti (Varsayılan = 3)" if lang == "Türkçe" else "Clinical cost of unnecessary intervention for healthy athlete (Default = 3)"
)

# ----------------------------------------------------
# AHP Weight Calculation & Display in Sidebar
# ----------------------------------------------------
ahp_weights = get_aggregated_ahp_weights(ahp_method)

st.sidebar.markdown("#### Seçilen AHP Ağırlıkları" if lang == "Türkçe" else "#### Selected AHP Weights")
ahp_w_df = pd.DataFrame({
    "Kriter" if lang == "Türkçe" else "Criterion": [crit_map[c] for c in CRITERIA],
    "Ağırlık" if lang == "Türkçe" else "Weight": ahp_weights
})
st.sidebar.dataframe(ahp_w_df.style.format({("Ağırlık" if lang == "Türkçe" else "Weight"): "{:.4f}"}))

# ----------------------------------------------------
# Data & Optimization Solves
# ----------------------------------------------------
raw_df = generate_raw_criteria(n_athletes=n_athletes, seed=seed)
df_owa = compute_owa_score(raw_df, owa_weights, ahp_weights)

# Add unique label to prevent Plotly summing/merging duplicated names!
df_owa["Athlete_Label"] = df_owa["Athlete_ID"] + " - " + df_owa["Name"]

# Run Solvers
df_m1, risk_m1 = solve_risk_prioritization(df_owa, M_capacity)
df_m2 = solve_threshold_classification(df_owa, T_threshold)
df_m3, cost_m3, analytical_th = solve_asymmetric_cost_optimization(df_owa, C_fn=c_fn, C_fp=c_fp)

# Merge Selections
df_merged = df_owa.copy()
df_merged["Model1_Selected"] = df_m1["Model1_Selected"]
df_merged["Model2_Selected"] = df_m2["Model2_Selected"]
df_merged["Risk_Class"] = df_m2["Risk_Class"]
df_merged["Model3_Selected"] = df_m3["Model3_Selected"]

# Tabs
if lang == "Türkçe":
    tab1, tab2, tab3 = st.tabs([
        "📋 Ham Veri & AHP/OWA Sonuçları", 
        "🎯 Karar Modelleri Karşılaştırması", 
        "📈 Duyarlılık ve Maliyet Ödünleşimi"
    ])
else:
    tab1, tab2, tab3 = st.tabs([
        "📋 Raw Data & AHP/OWA Results", 
        "🎯 Decision Models Comparison", 
        "📈 Sensitivity & Cost Trade-off"
    ])



# ----------------------------------------------------
# Tab 1: Ham Veriler ve AHP/OWA Aggregasyonu
# ----------------------------------------------------
with tab1:
    st.subheader("Biyomekanik Kriterler ve Risk Skorları" if lang == "Türkçe" else "Biomechanical Criteria and Risk Scores")
    if lang == "Türkçe":
        st.write(
            "Klinik uzmanların AHP ağırlıkları ve OWA birleştirme fonksiyonu kullanılarak, "
            "atletlerin **0-10 skalasında genel sakatlık risk skorları ($R_i$)** hesaplanmıştır."
        )
    else:
        st.write(
            "Using clinical experts' AHP weights and the OWA aggregation function, "
            "athletes' **overall injury risk scores ($R_i$) on a 0-10 scale** have been computed."
        )
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Style table
        table_df = df_merged.drop(columns=["Athlete_Label"]).copy()
        if lang == "Türkçe":
            table_df.columns = ["Atlet ID", "İsim"] + [CRITERIA_TR[c] for c in CRITERIA] + ["OWA Risk Skoru", "Model 1", "Model 2", "Risk Sınıfı", "Model 3"]
            show_cols = ["Atlet ID", "İsim"] + [CRITERIA_TR[c] for c in CRITERIA] + ["OWA Risk Skoru"]
            risk_score_col = "OWA Risk Skoru"
        else:
            table_df.columns = ["Athlete ID", "Name"] + [CRITERIA_EN[c] for c in CRITERIA] + ["OWA Risk Score", "Model 1", "Model 2", "Risk Class", "Model 3"]
            show_cols = ["Athlete ID", "Name"] + [CRITERIA_EN[c] for c in CRITERIA] + ["OWA Risk Score"]
            risk_score_col = "OWA Risk Score"
            
        st.dataframe(table_df[show_cols].style.background_gradient(subset=[risk_score_col], cmap="Reds").format({
            risk_score_col: "{:.2f}",
            **{crit_map[c]: "{:.2f}" for c in CRITERIA}
        }), height=400)
        
    with col2:
        # Plot AHP weights for selected scenario
        ahp_w_df_chart = ahp_w_df.copy()
        if lang == "English":
            ahp_w_df_chart.columns = ["Criterion", "Weight"]
            y_col, x_col = "Criterion", "Weight"
            ahp_title = f"AHP Criterion Weights ({ahp_choice_display})"
        else:
            y_col, x_col = "Kriter", "Ağırlık"
            ahp_title = f"AHP Kriter Ağırlıkları ({ahp_choice_display})"
            
        fig_ahp = px.bar(
            ahp_w_df_chart.sort_values(by=x_col, ascending=True),
            y=y_col,
            x=x_col,
            orientation='h',
            title=ahp_title,
            color=x_col,
            color_continuous_scale="Blues",
            height=400
        )
        fig_ahp.update_layout(showlegend=False)
        st.plotly_chart(fig_ahp, use_container_width=True)
        
    # Heatmap of criteria
    st.write("---")
    st.subheader("Detaylı Risk Kriterleri Matrisi (Heatmap)" if lang == "Türkçe" else "Detailed Risk Criteria Matrix (Heatmap)")
    
    heatmap_data = df_merged.set_index("Name")[CRITERIA]
    heatmap_data.columns = [crit_map[c] for c in CRITERIA]
    
    fig_heat = px.imshow(
        heatmap_data,
        color_continuous_scale="YlOrRd",
        labels=dict(
            x="Risk Kriteri" if lang == "Türkçe" else "Risk Criterion", 
            y="Atlet" if lang == "Türkçe" else "Athlete", 
            color="Risk Düzeyi" if lang == "Türkçe" else "Risk Level"
        ),
        aspect="auto",
        title="Atlet Kriter Bazlı Risk Seviyeleri (AHP ile ağırlıklandırılmamış ham riskler)" if lang == "Türkçe" else "Athlete Criterion-Based Risk Levels (Raw unweighted risks)",
        height=500
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ----------------------------------------------------
# Tab 2: Karar Modelleri Karşılaştırması
# ----------------------------------------------------
with tab2:
    st.subheader("Model Karar Çıktıları ve Karşılaştırmalar" if lang == "Türkçe" else "Model Decision Outputs and Comparisons")
    if lang == "Türkçe":
        st.write(
            "Klinisyenlerin anket cevapları ve kısıtları doğrultusunda optimize edilen üç karar modelinin "
            "atlet önceliklendirme çıktıları aşağıdadır."
        )
    else:
        st.write(
            "Below are the athlete prioritization outputs of the three decision models optimized in line with "
            "clinicians' survey responses and constraints."
        )
    
    # Compute total prioritized risks and costs
    m1_sel = df_merged["Model1_Selected"].sum()
    m3_sel = df_merged["Model3_Selected"].sum()
    m2_sel = df_merged["Model2_Selected"].sum()
    m1_risk_sum = df_merged[df_merged["Model1_Selected"] == 1]["OWA_Risk_Score"].sum()
    m2_risk_sum = df_merged[df_merged["Model2_Selected"] == 1]["OWA_Risk_Score"].sum()
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    # Text helper values for card labels and limits
    m1_label = "Model 1: Kapasite Kısıtlı" if lang == "Türkçe" else "Model 1: Capacity Constrained"
    m1_value = f"{m1_sel} Atlet Seçildi" if lang == "Türkçe" else f"{m1_sel} Athletes Selected"
    m1_sub = f"Toplam Risk: {m1_risk_sum:.2f}" if lang == "Türkçe" else f"Total Risk: {m1_risk_sum:.2f}"
    m1_constraint = f"Kısıt: Sabit Kapasite (M={M_capacity})" if lang == "Türkçe" else f"Constraint: Fixed Capacity (M={M_capacity})"
    
    m2_label = "Model 2: Klinik Sınıflandırma" if lang == "Türkçe" else "Model 2: Clinical Classification"
    m2_value = f"{m2_sel} Atlet Seçildi" if lang == "Türkçe" else f"{m2_sel} Athletes Selected"
    m2_sub = f"Toplam Risk: {m2_risk_sum:.2f}" if lang == "Türkçe" else f"Total Risk: {m2_risk_sum:.2f}"
    m2_constraint = f"Kısıt: Risk Eşiği (T={T_threshold:.2f})" if lang == "Türkçe" else f"Constraint: Risk Threshold (T={T_threshold:.2f})"
    
    m3_label = "Model 3: Asimetrik Klinik Maliyet" if lang == "Türkçe" else "Model 3: Asymmetric Clinical Cost"
    m3_value = f"{m3_sel} Atlet Seçildi" if lang == "Türkçe" else f"{m3_sel} Athletes Selected"
    m3_sub = f"Toplam Maliyet: {cost_m3:.2f}" if lang == "Türkçe" else f"Total Cost: {cost_m3:.2f}"
    m3_constraint = f"Maliyetler: C_FN={c_fn:.1f}, C_FP={c_fp:.1f}" if lang == "Türkçe" else f"Costs: C_FN={c_fn:.1f}, C_FP={c_fp:.1f}"
    
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{m1_label}</div>
            <div class="metric-value">{m1_value}</div>
            <div style="font-size:1.1rem; color:#10b981; font-weight:700; margin-top:5px;">{m1_sub}</div>
            <div style="font-size:0.8rem; color:#64748b; margin-top:5px;">{m1_constraint}</div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("🔍 Model 1 Mantığı ve Çözümü" if lang == "Türkçe" else "🔍 Model 1 Logic and Solution"):
            if lang == "Türkçe":
                st.markdown(rf"""
                **Uygulanma Amacı:** Klinik bütçe, personel veya zaman kısıtları altında izlenebilecek maksimum atlet sayısı sınırlı olduğunda, müdahale edilecek en kritik sporcuları matematiksel olarak önceliklendirmektir.
                """)
                st.markdown(r"""
                **Matematiksel Formül (LP/MILP):**
                $$ \max \sum_{i=1}^{N} R_i \cdot x_i \quad \text{s.t.} \quad \sum_{i=1}^{N} x_i \le M, \quad x_i \in \{0, 1\} $$
                """)
                st.markdown(f"""
                **Karar Yöntemi:** OWA risk skoruna ($R_i$) göre sporcular büyükten küçüğe sıralanır ve kapasite limiti ($M={M_capacity}$) kadar en yüksek riskli sporcu izlemeye seçilir.
                
                **Bulunan Sonuç:** PuLP çözücüsüyle en yüksek riskliler seçildi. Seçilen **{m1_sel}** sporcu ile önlenen toplam risk değeri **{m1_risk_sum:.2f}** birimdir.
                """)
            else:
                st.markdown(rf"""
                **Purpose:** To mathematically prioritize the most critical athletes to intervene when the maximum number of athletes that can be monitored is limited under clinical budget, personnel, or time constraints.
                """)
                st.markdown(r"""
                **Mathematical Formula (LP/MILP):**
                $$ \max \sum_{i=1}^{N} R_i \cdot x_i \quad \text{s.t.} \quad \sum_{i=1}^{N} x_i \le M, \quad x_i \in \{0, 1\} $$
                """)
                st.markdown(f"""
                **Decision Method:** Athletes are sorted in descending order according to their OWA risk score ($R_i$), and the highest risk athletes up to the capacity limit ($M={M_capacity}$) are selected for monitoring.
                
                **Results:** Highest risk athletes were selected using the PuLP solver. The total prevented risk value is **{m1_risk_sum:.2f}** units with **{m1_sel}** selected athletes.
                """)
        
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{m2_label}</div>
            <div class="metric-value">{m2_value}</div>
            <div style="font-size:1.1rem; color:#f59e0b; font-weight:700; margin-top:5px;">{m2_sub}</div>
            <div style="font-size:0.8rem; color:#64748b; margin-top:5px;">{m2_constraint}</div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("🔍 Model 2 Mantığı ve Çözümü" if lang == "Türkçe" else "🔍 Model 2 Logic and Solution"):
            if lang == "Türkçe":
                st.markdown(rf"""
                **Uygulanma Amacı:** Sabit bütçe sınırları yerine, klinik uzmanların belirlediği kritik risk düzeylerini (düşük/orta/yüksek) temel alarak sporcuları gruplandırmak ve risk eşiğini aşan her sporcuyu takip altına almaktır.
                """)
                st.markdown(r"""
                **Klinik Sınıflandırma Limitleri:**
                * **Düşük Risk ($R_i < 5.0$):** Haftalık 1 seans izleme.
                * **Orta Risk ($5.0 \le R_i < 6.0$):** Haftalık 3 seans izleme.
                * **Yüksek Risk ($R_i \ge 6.0$):** Klinik sevk ve günlük izleme.
                """)
                st.markdown(f"""
                **Karar Yöntemi:** Risk skoru, belirlediğiniz risk eşik değerini ($T={T_threshold:.2f}$) aşan tüm sporcular izleme programına dahil edilir.
                
                **Bulunan Sonuç:** Eşiği aşan **{m2_sel}** sporcu seçilerek **{m2_risk_sum:.2f}** birim toplam risk kapsama altına alındı.
                """)
            else:
                st.markdown(rf"""
                **Purpose:** Instead of fixed budget limits, grouping athletes based on clinical risk levels (low/medium/high) set by clinical experts and tracking every athlete who exceeds the risk threshold.
                """)
                st.markdown(r"""
                **Clinical Classification Limits:**
                * **Low Risk ($R_i < 5.0$):** Weekly 1 session of monitoring.
                * **Medium Risk ($5.0 \le R_i < 6.0$):** Weekly 3 sessions of monitoring.
                * **High Risk ($R_i \ge 6.0$):** Clinical referral and daily monitoring.
                """)
                st.markdown(f"""
                **Decision Method:** All athletes whose risk score exceeds the risk threshold value ($T={T_threshold:.2f}$) are included in the monitoring program.
                
                **Results:** **{m2_sel}** athletes exceeding the threshold were selected, covering a total risk of **{m2_risk_sum:.2f}** units.
                """)
        
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{m3_label}</div>
            <div class="metric-value">{m3_value}</div>
            <div style="font-size:1.1rem; color:#dc2626; font-weight:700; margin-top:5px;">{m3_sub}</div>
            <div style="font-size:0.8rem; color:#64748b; margin-top:5px;">{m3_constraint}</div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("🔍 Model 3 Mantığı ve Çözümü" if lang == "Türkçe" else "🔍 Model 3 Logic and Solution"):
            if lang == "Türkçe":
                st.markdown(rf"""
                **Uygulanma Amacı:** Bir sporcunun sakatlanmasını kaçırmanın klinik tehlikesi ile sağlam sporcuya gereksiz müdahalede bulunmanın getirdiği iş gücü/ekipman kaybını asimetrik olarak dengeleyerek toplam beklenen kaybı en aza indirmektir.
                """)
                st.markdown(r"""
                **Matematiksel Formül (Kayıp Minimizasyonu):**
                $$ \min \sum_{i=1}^{N} \left[ C_{FN} \cdot \left(\frac{R_i}{10}\right) \cdot (1 - x_i) + C_{FP} \cdot \left(1 - \frac{R_i}{10}\right) \cdot x_i \right] \quad x_i \in \{0, 1\} $$
                """)
                st.markdown(f"""
                **Karar Yöntemi (Analitik Eşik):** Yanlış Negatif ($C_{{FN}}={c_fn:.1f}$) ve Yanlış Pozitif ($C_{{FP}}={c_fp:.1f}$) maliyetlerine göre optimal karar eşiği:
                $$ T_{{opt}} = 10 \\cdot \\frac{{C_{{FP}}}}{{C_{{FN}} + C_{{FP}}}} = {analytical_th:.2f} $$
                OWA risk skoru $R_i > {analytical_th:.2f}$ olan tüm sporcular seçilir.
                
                **Bulunan Sonuç:** Toplam beklenen klinik maliyet **{cost_m3:.2f}** birime minimize edildi.
                """)
            else:
                st.markdown(rf"""
                **Purpose:** To minimize the total expected loss by asymmetrically balancing the clinical danger of missing an athlete's injury (False Negative) with the loss of labor/equipment caused by unnecessary intervention on a healthy athlete (False Positive).
                """)
                st.markdown(r"""
                **Mathematical Formula (Loss Minimization):**
                $$ \min \sum_{i=1}^{N} \left[ C_{FN} \cdot \left(\frac{R_i}{10}\right) \cdot (1 - x_i) + C_{FP} \cdot \left(1 - \frac{R_i}{10}\right) \cdot x_i \right] \quad x_i \in \{0, 1\} $$
                """)
                st.markdown(f"""
                **Decision Method (Analytical Threshold):** Optimal decision threshold according to False Negative ($C_{{FN}}={c_fn:.1f}$) and False Positive ($C_{{FP}}={c_fp:.1f}$) costs:
                $$ T_{{opt}} = 10 \\cdot \\frac{{C_{{FP}}}}{{C_{{FN}} + C_{{FP}}}} = {analytical_th:.2f} $$
                All athletes with OWA risk score $R_i > {analytical_th:.2f}$ are selected.
                
                **Results:** Total expected clinical cost was minimized to **{cost_m3:.2f}** units.
                """)
        
    st.write("<br>", unsafe_allow_html=True)
    st.subheader("Atlet Bazlı Karar Karşılaştırma Matrisi" if lang == "Türkçe" else "Athlete-Based Decision Comparison Matrix")
    
    comp_df = df_merged.copy()
    
    # Translate row texts
    intervene_label = "✅ Müdahale Et" if lang == "Türkçe" else "✅ Intervene"
    monitor_label = "❌ İzleme" if lang == "Türkçe" else "❌ Monitor Only"
    intervene_with_class = "✅ Müdahale" if lang == "Türkçe" else "✅ Intervene"
    monitor_with_class = "❌ İzleme" if lang == "Türkçe" else "❌ Monitor Only"
    
    # Map Risk Class
    comp_df["Risk_Class_Display"] = comp_df["Risk_Class"].map(lambda x: RISK_CLASS_MAP[x][lang])
    
    comp_df["Model 1 (M)"] = comp_df["Model1_Selected"].apply(lambda val: intervene_label if val == 1 else monitor_label)
    comp_df["Model 2 (Klinik Sınıf)"] = comp_df.apply(
        lambda row: f"{intervene_with_class} ({row['Risk_Class_Display']})" if row["Model2_Selected"] == 1 else f"{monitor_with_class} ({row['Risk_Class_Display']})", axis=1
    )
    comp_df["Model 3 (Maliyet)"] = comp_df["Model3_Selected"].apply(lambda val: intervene_label if val == 1 else monitor_label)
    
    comp_df = comp_df[["Athlete_ID", "Name", "OWA_Risk_Score", "Model 1 (M)", "Model 2 (Klinik Sınıf)", "Model 3 (Maliyet)"]]
    if lang == "Türkçe":
        comp_df.columns = ["Atlet ID", "İsim", "OWA Risk Skoru (0-10)", "Model 1 (M Kapasiteli)", "Model 2 (Eşikli Sınıflandırma)", "Model 3 (Asimetrik Kayıp)"]
    else:
        comp_df.columns = ["Athlete ID", "Name", "OWA Risk Score (0-10)", "Model 1 (M Capacity)", "Model 2 (Threshold Classification)", "Model 3 (Asymmetric Loss)"]
        
    st.dataframe(comp_df.style.format({("OWA Risk Skoru (0-10)" if lang == "Türkçe" else "OWA Risk Score (0-10)"): "{:.2f}"}), height=450)
    
    # AHP Scenarios comparison bar chart
    st.write("---")
    st.subheader("Farklı AHP Uzman Birleştirme Senaryolarına Göre Risk Kararlılığı" if lang == "Türkçe" else "Athlete Risk Stability Across AHP Aggregation Scenarios")
    if lang == "Türkçe":
        st.write(
            "Aşağıdaki grafik, tüm atletlerin OWA risk skorlarının 5 farklı AHP uzman aggregasyon metoduna göre değişimini gösterir. "
            "Bu analiz, karar vericilere uzman fikir ayrılıklarının atlet risk sıralamalarını ne kadar etkilediğini görme fırsatı sunar."
        )
    else:
        st.write(
            "The chart below shows how each athlete's OWA risk score changes across the 5 different AHP expert aggregation methods. "
            "This analysis allows decision makers to see how expert disagreements affect athlete risk rankings."
        )
    
    # Calculate scores under all 5 scenarios
    scenarios = ["Neutral / Average", "Optimistic / Max", "Pessimistic / Min", "CR-Based", "Confidence-Based"]
    comparison_scores = []
    
    for scen in scenarios:
        w = get_aggregated_ahp_weights(scen)
        scen_owa_df = compute_owa_score(raw_df, owa_weights, w)
        scen_display = AHP_METHOD_MAP[scen][lang]
        for idx, row in scen_owa_df.iterrows():
            comparison_scores.append({
                "Athlete": row["Athlete_ID"] + " - " + row["Name"],
                "Score": row["OWA_Risk_Score"],
                "Senaryo": scen_display
            })
            
    df_compare = pd.DataFrame(comparison_scores)
    
    # Sort athletes by Neutral score first to keep plot clean
    neutral_order = df_merged.sort_values(by="OWA_Risk_Score", ascending=False)["Athlete_Label"].tolist()
    
    fig_compare = px.bar(
        df_compare,
        x="Athlete",
        y="Score",
        color="Senaryo",
        barmode="group",
        category_orders={"Athlete": neutral_order},
        title="AHP Senaryolarına Göre Atlet Risk Skorlarının Dağılımı" if lang == "Türkçe" else "Distribution of Athlete Risk Scores by AHP Scenarios",
        labels={
            "Athlete": "Atlet (ID - İsim)" if lang == "Türkçe" else "Athlete (ID - Name)", 
            "Score": "OWA Risk Skoru (0-10)" if lang == "Türkçe" else "OWA Risk Score (0-10)", 
            "Senaryo": "AHP Aggregasyonu" if lang == "Türkçe" else "AHP Aggregation"
        },
        height=500
    )
    fig_compare.update_layout(barmode='group', xaxis_tickangle=-45)
    st.plotly_chart(fig_compare, use_container_width=True)

# ----------------------------------------------------
# Tab 3: Duyarlılık ve Maliyet Ödünleşimi
# ----------------------------------------------------
with tab3:
    st.subheader("Asimetrik Klinik Maliyet Duyarlılık Analizi" if lang == "Türkçe" else "Asymmetric Clinical Cost Sensitivity Analysis")
    if lang == "Türkçe":
        st.write(
            "Klinik uygulamada, sakatlığı kaçırmanın (False Negative) ve sağlıklı bir sporcuya gereksiz zaman harcamanın "
            "(False Positive) klinik ağırlıkları değiştikçe, sistemin otomatik risk önceliklendirme eşiği de dinamik olarak kayar. "
            "Bu tabda maliyet oranlarının kararlılığa etkisi incelenmektedir."
        )
    else:
        st.write(
            "In clinical practice, as the clinical weights of missing an injury (False Negative) and wasting unnecessary time "
            "on a healthy athlete (False Positive) change, the system's automatic risk prioritization threshold also shifts dynamically. "
            "This tab analyzes the effect of cost ratios on decision stability."
        )
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("#### Cost Ratio (C_FN / C_FP) vs. Karar Eşiği" if lang == "Türkçe" else "#### Cost Ratio (C_FN / C_FP) vs. Decision Threshold")
        if lang == "Türkçe":
            st.write(
                "C_FP = 3'te sabit tutularak, C_FN (Sakatlığı Kaçırma Maliyeti) 1.0'dan 27.0'ye yükseltildiğinde "
                "karar modelinin optimal **seçim eşik değerinin ($T_{opt}$)** nasıl düştüğü gösterilmektedir. "
                "Maliyet oranı yükseldikçe, model daha korumacı (muhafazakar) hale gelerek daha düşük riskteki atletleri de seçer."
            )
        else:
            st.write(
                "Keeping C_FP = 3 constant, as C_FN (Injury Missing Cost) increases from 1.0 to 27.0, the optimal "
                "**selection threshold value ($T_{opt}$)** of the decision model decreases. "
                "As the cost ratio increases, the model becomes more protective (conservative) and selects lower-risk athletes as well."
            )
        
        sens_costs_df = run_sensitivity_analysis_costs(df_owa, cost_ratio_steps=40)
        
        fig_th = px.line(
            sens_costs_df,
            x="Ratio_FN_FP",
            y="Analytical_Threshold",
            markers=True,
            title="Maliyet Oranı (C_FN / C_FP) vs. Karar Eşiği" if lang == "Türkçe" else "Cost Ratio (C_FN / C_FP) vs. Decision Threshold",
            labels={
                "Ratio_FN_FP": "Klinik Hata Maliyet Oranı (C_FN / C_FP)" if lang == "Türkçe" else "Clinical Error Cost Ratio (C_FN / C_FP)", 
                "Analytical_Threshold": "Optimal Seçim Eşik Skoru (0-10)" if lang == "Türkçe" else "Optimal Selection Threshold Score (0-10)"
            },
            height=400
        )
        fig_th.update_traces(line_color="#dc2626")
        
        # Add current slider ratio indicator
        current_ratio = c_fn / c_fp
        fig_th.add_vline(
            x=current_ratio, 
            line_dash="dash", 
            line_color="blue", 
            annotation_text=f"Şu anki Oran ({current_ratio:.2f})" if lang == "Türkçe" else f"Current Ratio ({current_ratio:.2f})"
        )
        st.plotly_chart(fig_th, use_container_width=True)
        
    with col_s2:
        st.markdown("#### Maliyet Oranına Göre Önceliklendirilen Atlet Sayısı" if lang == "Türkçe" else "#### Number of Prioritized Athletes by Cost Ratio")
        if lang == "Türkçe":
            st.write(
                "Yine klinik hata maliyeti oranı arttıkça, izleme kapsamına (müdahaleye) alınan toplam atlet sayısının "
                "nasıl arttığı gösterilmektedir. Bu eğri karar vericiye kaynak planlaması açısından yol gösterir."
            )
        else:
            st.write(
                "Similarly, as the clinical error cost ratio increases, the total number of athletes included in monitoring "
                "(intervention) increases. This curve guides the decision maker in terms of resource planning."
            )
        
        fig_cnt = px.line(
            sens_costs_df,
            x="Ratio_FN_FP",
            y="Selected_Count",
            markers=True,
            title="Maliyet Oranı (C_FN / C_FP) vs. Müdahale Sayısı" if lang == "Türkçe" else "Cost Ratio (C_FN / C_FP) vs. Number of Interventions",
            labels={
                "Ratio_FN_FP": "Klinik Hata Maliyet Oranı (C_FN / C_FP)" if lang == "Türkçe" else "Clinical Error Cost Ratio (C_FN / C_FP)", 
                "Selected_Count": "Önceliklendirilen Atlet Sayısı" if lang == "Türkçe" else "Number of Prioritized Athletes"
            },
            height=400
        )
        fig_cnt.update_traces(line_color="#2563eb")
        fig_cnt.add_vline(
            x=current_ratio, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"Şu anki Seçim ({m3_sel} Atlet)" if lang == "Türkçe" else f"Current Selection ({m3_sel} Athletes)"
        )
        st.plotly_chart(fig_cnt, use_container_width=True)
        
    # Asymmetric expected cost mapping
    st.write("---")
    st.subheader("Model 3: Klinik Hata Kararlılık Matrisi" if lang == "Türkçe" else "Model 3: Clinical Error Stability Matrix")
    if lang == "Türkçe":
        st.write(
            "Farklı C_FN/C_FP oranları altında her atletin sakatlık müdahalesine ne kadar kararlı bir şekilde seçildiğini test ediyoruz. "
            "Eğer bir atlet maliyet oranı en düşük durumdayken dahi seçiliyorsa, sakatlık riski son derece kritik düzeydedir."
        )
    else:
        st.write(
            "We test how stably each athlete is selected for injury intervention under different C_FN/C_FP ratios. "
            "If an athlete is selected even when the cost ratio is at its lowest, their injury risk is at an extremely critical level."
        )
    
    # Calculate decision frequencies across 40 cost ratio scenarios
    cfn_vals = np.linspace(1.0, 15.0, 40)
    stability_matrix = np.zeros(len(df_owa))
    
    for cf in cfn_vals:
        res_df, _, _ = solve_asymmetric_cost_optimization(df_owa, C_fn=cf, C_fp=3.0)
        stability_matrix += res_df["Model3_Selected"].values
        
    df_stab_cost = df_owa.copy()
    df_stab_cost["Stability_Pct"] = (stability_matrix / len(cfn_vals)) * 100
    df_stab_cost = df_stab_cost.sort_values(by="Stability_Pct", ascending=True)
    
    fig_stab_cost = px.bar(
        df_stab_cost,
        y="Athlete_Label",
        x="Stability_Pct",
        orientation='h',
        title="Maliyet Değişimlerine Göre Atlet Önceliklendirme Kararlılığı (Karar Frekansı %)" if lang == "Türkçe" else "Athlete Prioritization Stability under Cost Variations (Decision Frequency %)",
        labels={
            "Athlete_Label": "Atlet (ID - İsim)" if lang == "Türkçe" else "Athlete (ID - Name)", 
            "Stability_Pct": "Seçilme Kararlılığı (%)" if lang == "Türkçe" else "Selection Stability (%)"
        },
        color="Stability_Pct",
        color_continuous_scale="Viridis",
        height=500
    )
    st.plotly_chart(fig_stab_cost, use_container_width=True)

    # ----------------------------------------------------
    # Model 1 Capacity & Shadow Price Sensitivity Section
    # ----------------------------------------------------
    st.write("---")
    st.subheader("📊 Model 1: Kaynak Kapasitesi (M) ve Gölge Fiyat (Shadow Price) Analizi" if lang == "Türkçe" else "📊 Model 1: Resource Capacity (M) and Shadow Price Analysis")
    if lang == "Türkçe":
        st.write(
            "Kapasite kısıtı $M$'in gölge fiyatı (shadow price), sisteme eklenen bir adet ek takip/müdahale yuvasının "
            "sağladığı marjinal sakatlık riski azaltımını ölçer. "
            "Aşağıdaki grafiklerde, kapasite arttıkça kapsanan toplam riskin artışı (Sol) ve eklenen her yeni kapasite biriminin "
            "getirdiği marjinal faydanın (Sağ - Gölge Fiyat) azalan seyri yan yana gösterilmektedir."
        )
    else:
        st.write(
            "The shadow price of the capacity constraint $M$ measures the marginal reduction in injury risk provided by adding "
            "one additional monitoring/intervention slot to the system. The charts side-by-side show the increase in total "
            "covered risk as capacity increases (Left) and the decreasing marginal benefit of each new unit of capacity (Right - Shadow Price)."
        )
    
    # Calculate shadow prices and cumulative risk
    from optimization_models import compute_shadow_prices
    shadow_prices = compute_shadow_prices(df_owa)
    
    m_values = list(range(1, len(df_owa) + 1))
    risk_covered_list = []
    for m in m_values:
        _, risk_cov = solve_risk_prioritization(df_owa, m)
        risk_covered_list.append(risk_cov)
        
    df_m_sens = pd.DataFrame({
        "Kapasite_M": m_values,
        "Kapsanan_Toplam_Risk": risk_covered_list,
        "Golge_Fiyat": shadow_prices
    })
    
    # Current shadow price at M_capacity
    # Marginal benefit of the NEXT slot: index M_capacity in sorted list
    if M_capacity < len(shadow_prices):
        current_shadow_price = shadow_prices[M_capacity]
    else:
        current_shadow_price = 0.0
        
    # Render shadow price metric card
    shadow_card_title = f"M={M_capacity} Kapasitesindeki Gölge Fiyat (Shadow Price)" if lang == "Türkçe" else f"Shadow Price at M={M_capacity} Capacity"
    shadow_card_val = f"{current_shadow_price:.2f} Risk Azaltımı" if lang == "Türkçe" else f"{current_shadow_price:.2f} Risk Reduction"
    if lang == "Türkçe":
        shadow_card_desc = f"Kapasiteyi {M_capacity}'den {M_capacity+1}'e çıkarmak, sisteme ek olarak <b>{current_shadow_price:.2f}</b> birim sakatlık riski azaltımı kazandırır."
    else:
        shadow_card_desc = f"Increasing capacity from {M_capacity} to {M_capacity+1} provides an additional <b>{current_shadow_price:.2f}</b> units of injury risk reduction to the system."

    st.markdown(f"""
    <div class="metric-card" style="margin-bottom: 20px;">
        <div class="metric-label">{shadow_card_title}</div>
        <div class="metric-value">{shadow_card_val}</div>
        <div style="font-size:0.9rem; color:#1e3a8a; font-weight:600;">
            {shadow_card_desc}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_cap1, col_cap2 = st.columns(2)
    
    with col_cap1:
        fig_m_sens = px.line(
            df_m_sens,
            x="Kapasite_M",
            y="Kapsanan_Toplam_Risk",
            markers=True,
            title="Kapasite (M) vs. Kapsanan Toplam Risk (Toplam Fayda)" if lang == "Türkçe" else "Capacity (M) vs. Total Covered Risk (Total Benefit)",
            labels={
                "Kapasite_M": "İzlenebilecek Atlet Sayısı (M)" if lang == "Türkçe" else "Number of Athletes to Monitor (M)", 
                "Kapsanan_Toplam_Risk": "Kapsanan Toplam Sakatlık Riski" if lang == "Türkçe" else "Total Covered Injury Risk"
            },
            height=400
        )
        fig_m_sens.update_traces(line_color="#2563eb", marker=dict(size=6))
        fig_m_sens.add_vline(
            x=M_capacity, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"M={M_capacity} (Risk={risk_covered_list[M_capacity-1]:.2f})"
        )
        st.plotly_chart(fig_m_sens, use_container_width=True)
        
    with col_cap2:
        fig_shadow = px.bar(
            df_m_sens,
            x="Kapasite_M",
            y="Golge_Fiyat",
            title="Kapasite (M) vs. Gölge Fiyatı (Marjinal Fayda)" if lang == "Türkçe" else "Capacity (M) vs. Shadow Price (Marginal Benefit)",
            labels={
                "Kapasite_M": "Kapasite (M)" if lang == "Türkçe" else "Capacity (M)", 
                "Golge_Fiyat": "Gölge Fiyat (Risk Azaltım Değeri)" if lang == "Türkçe" else "Shadow Price (Risk Reduction Value)"
            },
            height=400
        )
        fig_shadow.update_traces(marker_color="#10b981")
        fig_shadow.add_vline(
            x=M_capacity, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"Şu anki Gölge Fiyat ({current_shadow_price:.2f})" if lang == "Türkçe" else f"Current Shadow Price ({current_shadow_price:.2f})"
        )
        st.plotly_chart(fig_shadow, use_container_width=True)
