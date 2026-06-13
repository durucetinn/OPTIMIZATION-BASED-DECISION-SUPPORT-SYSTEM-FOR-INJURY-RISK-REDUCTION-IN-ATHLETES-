import pulp
import pandas as pd
import numpy as np

def solve_risk_prioritization(df, M):
    """
    Model 1: Risk Prioritization Model (Kapasite Kısıtlı Önceliklendirme)
    
    Objective: Select M athletes to maximize total prioritized risk.
    Maximize sum(R_i * x_i)
    Subject to:
      sum(x_i) <= M
      x_i in {0, 1}
    """
    assert "OWA_Risk_Score" in df.columns, "Dataframe must contain 'OWA_Risk_Score'."
    M = min(max(0, int(M)), len(df))
    
    prob = pulp.LpProblem("Risk_Prioritization", pulp.LpMaximize)
    athlete_ids = df.index.tolist()
    x = pulp.LpVariable.dicts("Select", athlete_ids, cat=pulp.LpBinary)
    
    # Objective function
    prob += pulp.lpSum(df.loc[i, "OWA_Risk_Score"] * x[i] for i in athlete_ids), "Total_Risk_Covered"
    
    # Resource constraint
    prob += pulp.lpSum(x[i] for i in athlete_ids) <= M, "Resource_Constraint"
    
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    result_df = df.copy()
    selected = [int(pulp.value(x[i])) for i in athlete_ids]
    result_df["Model1_Selected"] = selected
    
    total_risk_covered = pulp.value(prob.objective)
    return result_df, total_risk_covered

def solve_threshold_classification(df, T):
    """
    Model 2: Threshold-Based Risk Classification
    
    Classifies athletes into 3 risk levels based on clinical expert answers:
    - Low Risk (Düşük): Score < 5.0 (0)
    - Medium Risk (Orta): 5.0 <= Score < 6.0 (1)
    - High Risk (Yüksek): Score >= 6.0 (2)
    
    Also provides a binary selection based on user-defined threshold T.
    """
    assert "OWA_Risk_Score" in df.columns, "Dataframe must contain 'OWA_Risk_Score'."
    
    result_df = df.copy()
    
    # Three-tier classification
    risk_classes = []
    for score in df["OWA_Risk_Score"]:
        if score < 5.0:
            risk_classes.append("Düşük Risk")
        elif score < 6.0:
            risk_classes.append("Orta Risk")
        else:
            risk_classes.append("Yüksek Risk")
            
    result_df["Risk_Class"] = risk_classes
    # Binary selection based on threshold T
    result_df["Model2_Selected"] = (df["OWA_Risk_Score"] >= T).astype(int)
    
    return result_df

def solve_asymmetric_cost_optimization(df, C_fn=9, C_fp=3):
    """
    Model 3: Asymmetric Clinical Cost Optimization (Asimetrik Hata Maliyeti Modeli)
    
    Expected Loss for each athlete:
    - False Negative (missing a high-risk athlete): C_fn * (R_i/10) * (1 - x_i)
    - False Positive (unnecessary intervention): C_fp * (1 - R_i/10) * x_i
    
    Objective:
    Minimize sum( C_fn * (R_i/10) * (1 - x_i) + C_fp * (1 - R_i/10) * x_i )
    Subject to:
      x_i in {0, 1}
      
    Analytical selection threshold is:
    R_i > 10 * C_fp / (C_fn + C_fp)
    For C_fn=9, C_fp=3, the threshold is 10 * 3/12 = 2.5.
    """
    assert "OWA_Risk_Score" in df.columns, "Dataframe must contain 'OWA_Risk_Score'."
    assert C_fn >= 0 and C_fp >= 0, "Costs must be non-negative."
    
    prob = pulp.LpProblem("Asymmetric_Cost_Minimization", pulp.LpMinimize)
    athlete_ids = df.index.tolist()
    x = pulp.LpVariable.dicts("Select", athlete_ids, cat=pulp.LpBinary)
    
    # Objective
    prob += pulp.lpSum(
        C_fn * (df.loc[i, "OWA_Risk_Score"] / 10.0) * (1 - x[i]) +
        C_fp * (1.0 - df.loc[i, "OWA_Risk_Score"] / 10.0) * x[i]
        for i in athlete_ids
    ), "Total_Clinical_Loss"
    
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    result_df = df.copy()
    selected = [int(pulp.value(x[i])) for i in athlete_ids]
    result_df["Model3_Selected"] = selected
    
    total_cost = pulp.value(prob.objective)
    analytical_threshold = 10.0 * C_fp / (C_fn + C_fp) if (C_fn + C_fp) > 0 else 0.0
    
    return result_df, total_cost, analytical_threshold

def run_sensitivity_analysis_costs(df, cost_ratio_steps=30):
    """
    Varies the ratio of C_fn / C_fp and tracks selection counts and threshold changes.
    """
    # Fix C_fp = 3, vary C_fn from 1 to 27 (so C_fn/C_fp ratio goes from 1/3 to 9)
    c_fp = 3.0
    c_fn_values = np.linspace(1.0, 27.0, cost_ratio_steps)
    results = []
    
    for c_fn in c_fn_values:
        res_df, total_loss, th = solve_asymmetric_cost_optimization(df, C_fn=c_fn, C_fp=c_fp)
        selected_count = res_df["Model3_Selected"].sum()
        
        results.append({
            "C_FN": c_fn,
            "C_FP": c_fp,
            "Ratio_FN_FP": c_fn / c_fp,
            "Analytical_Threshold": th,
            "Selected_Count": selected_count,
            "Expected_Loss": total_loss
        })
        
    return pd.DataFrame(results)
def compute_shadow_prices(df):
    """
    Computes the empirical shadow price for each capacity level M (1 to N).
    The shadow price is the marginal risk reduction achieved by adding the M-th slot of capacity.
    Since we select athletes greedily by OWA risk score, the shadow price at capacity M
    is the OWA risk score of the M-th athlete when sorted in descending order of risk.
    """
    assert "OWA_Risk_Score" in df.columns, "Dataframe must contain 'OWA_Risk_Score'."
    # Sort descending
    sorted_scores = df["OWA_Risk_Score"].sort_values(ascending=False).tolist()
    return sorted_scores

if __name__ == "__main__":
    from data_loader import generate_raw_criteria, get_aggregated_ahp_weights, compute_owa_score
    
    df = generate_raw_criteria(n_athletes=10)
    ahp_w = get_aggregated_ahp_weights("Neutral / Average")
    owa_w = np.array([0.35, 0.25, 0.20, 0.10, 0.07, 0.03])
    df = compute_owa_score(df, owa_w, ahp_w)
    
    print("--- Model 2: Three-Tier Risk Classification Test ---")
    df_m2 = solve_threshold_classification(df, T=5.0)
    print(df_m2[["Athlete_ID", "Name", "OWA_Risk_Score", "Risk_Class", "Model2_Selected"]])
    
    print("\n--- Model 3: Asymmetric Clinical Cost Optimization Test (C_fn=9, C_fp=3) ---")
    df_m3, cost, th = solve_asymmetric_cost_optimization(df, C_fn=9, C_fp=3)
    print(df_m3[df_m3["Model3_Selected"] == 1][["Athlete_ID", "Name", "OWA_Risk_Score"]])
    print(f"Total Loss: {cost:.4f}, Analytical Threshold: {th:.2f}")
