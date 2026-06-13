import unittest
import numpy as np
import pandas as pd
from data_loader import generate_raw_criteria, get_aggregated_ahp_weights, compute_owa_score
from optimization_models import (
    solve_risk_prioritization,
    solve_threshold_classification,
    solve_asymmetric_cost_optimization
)

class TestOptimizationModels(unittest.TestCase):
    
    def setUp(self):
        # Generate stable test data
        self.df = generate_raw_criteria(n_athletes=30, seed=100)
        self.ahp_w = get_aggregated_ahp_weights("Confidence-Based")
        # OWA weights: prioritize the worst criteria (conservative risk aggregation)
        self.owa_w = np.array([0.30, 0.25, 0.20, 0.12, 0.08, 0.05])
        self.df = compute_owa_score(self.df, self.owa_w, self.ahp_w)
        
    def test_model1_resource_constraint(self):
        """
        Verify Model 1:
        1. Selects at most M athletes.
        2. Selects the M athletes with the highest risk scores.
        """
        M = 8
        result_df, total_risk = solve_risk_prioritization(self.df, M=M)
        
        # 1. Resource capacity limit holds
        selected_count = result_df["Model1_Selected"].sum()
        self.assertTrue(selected_count <= M, f"Selected {selected_count} athletes, limit was {M}")
        
        # 2. Verify that selection is greedy-optimal (highest risk first)
        sorted_df = result_df.sort_values(by="OWA_Risk_Score", ascending=False)
        top_m_selected = sorted_df.iloc[:M]["Model1_Selected"].tolist()
        rest_selected = sorted_df.iloc[M:][["Athlete_ID", "Model1_Selected"]]
        
        for val in top_m_selected:
            self.assertEqual(val, 1, "A high-risk athlete in the top M was not prioritized.")
            
        for val in rest_selected["Model1_Selected"]:
            self.assertEqual(val, 0, "An athlete outside the top M was prioritized.")
            
    def test_model2_three_tier_classification(self):
        """
        Verify Model 2 three-tier risk classification:
        - Score < 5.0 -> Düşük Risk
        - 5.0 <= Score < 6.0 -> Orta Risk
        - Score >= 6.0 -> Yüksek Risk
        """
        result_df = solve_threshold_classification(self.df, T=5.5)
        
        for idx, row in result_df.iterrows():
            score = row["OWA_Risk_Score"]
            risk_class = row["Risk_Class"]
            
            if score < 5.0:
                self.assertEqual(risk_class, "Düşük Risk", f"Score {score} < 5.0 but class is {risk_class}")
            elif score < 6.0:
                self.assertEqual(risk_class, "Orta Risk", f"Score {score} is in [5,6) but class is {risk_class}")
            else:
                self.assertEqual(risk_class, "Yüksek Risk", f"Score {score} >= 6.0 but class is {risk_class}")

    def test_model3_asymmetric_cost_thresholds(self):
        """
        Verify Model 3 asymmetric clinical cost selection rule:
        Selected iff R_i > 10 * C_fp / (C_fn + C_fp).
        We test multiple clinical cost scenarios:
        1. C_fn=9, C_fp=3  => threshold = 2.50
        2. C_fn=5, C_fp=5  => threshold = 5.00
        3. C_fn=2, C_fp=8  => threshold = 8.00
        """
        scenarios = [
            {"C_fn": 9, "C_fp": 3, "expected_th": 2.50},
            {"C_fn": 5, "C_fp": 5, "expected_th": 5.00},
            {"C_fn": 2, "C_fp": 8, "expected_th": 8.00}
        ]
        
        for scene in scenarios:
            c_fn = scene["C_fn"]
            c_fp = scene["C_fp"]
            expected_th = scene["expected_th"]
            
            result_df, _, calc_th = solve_asymmetric_cost_optimization(self.df, C_fn=c_fn, C_fp=c_fp)
            self.assertAlmostEqual(calc_th, expected_th, places=4, msg="Calculated threshold is incorrect.")
            
            for idx, row in result_df.iterrows():
                score = row["OWA_Risk_Score"]
                selected = row["Model3_Selected"]
                
                # Check with float tolerance
                if score > (expected_th + 1e-7):
                    self.assertEqual(selected, 1, f"Score {score} > {expected_th} but not selected for costs {c_fn}:{c_fp}")
                elif score < (expected_th - 1e-7):
                    self.assertEqual(selected, 0, f"Score {score} < {expected_th} but was selected for costs {c_fn}:{c_fp}")

if __name__ == "__main__":
    unittest.main()
