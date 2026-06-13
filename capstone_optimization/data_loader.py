import numpy as np
import pandas as pd

# Define the 6 criteria based on the expert AHP sheets
CRITERIA = [
    "Ankle_Foot_Mechanics",
    "Knee_Mechanics",
    "Hip_Mechanics",
    "Trunk_Control",
    "Stance_Lower_Extremity",
    "Overall_Landing_Quality"
]

CRITERIA_TR = {
    "Ankle_Foot_Mechanics": "Ayak Bileği ve Ayak Mekaniği",
    "Knee_Mechanics": "Diz Mekaniği",
    "Hip_Mechanics": "Kalça Mekaniği",
    "Trunk_Control": "Gövde Kontrolü",
    "Stance_Lower_Extremity": "Duruş ve Alt Ekstremite",
    "Overall_Landing_Quality": "Genel İniş Kalitesi / Motor Kontrol"
}

# Exact AHP weights from the 4 experts
EXPERT_AHP_WEIGHTS = {
    "Cem (Expert-1)": np.array([0.099198285, 0.290934157, 0.046720632, 0.077700243, 0.090492931, 0.394953752]),
    "Esma (Expert-2)": np.array([0.031994422, 0.517113687, 0.036466527, 0.099020280, 0.118997706, 0.196407377]),
    "Görkem (Expert-3)": np.array([0.104662000, 0.257287227, 0.065066066, 0.116434927, 0.087051608, 0.369493971]),
    "Sabriye (Expert-4)": np.array([0.135970368, 0.310420835, 0.050774231, 0.178273041, 0.178273041, 0.146288484])
}

# CR-Based expert weights (inverse of CR, normalized)
CR_EXPERT_WEIGHTS = np.array([0.396282485, 0.172544537, 0.215598115, 0.215574863])

# Confidence-Weighted expert weights (derived from Z-scores of expert comparisons)
CONF_EXPERT_WEIGHTS = np.array([0.064313229, 0.203759289, 0.098895786, 0.633031777])

def get_aggregated_ahp_weights(method="Neutral / Average"):
    """
    Computes aggregated weights for the 6 criteria based on AHP scenario sheets.
    """
    experts = list(EXPERT_AHP_WEIGHTS.values())
    
    if method == "Neutral / Average":
        # Simple arithmetic mean of experts
        w = np.mean(experts, axis=0)
    elif method == "Optimistic / Max":
        # Max weight across experts for each criterion, then normalized
        w = np.max(experts, axis=0)
    elif method == "Pessimistic / Min":
        # Min weight across experts for each criterion, then normalized
        w = np.min(experts, axis=0)
    elif method == "CR-Based":
        # Weighted by Consistency Ratio reliability
        w = np.zeros(6)
        for idx, (name, weights) in enumerate(EXPERT_AHP_WEIGHTS.items()):
            w += CR_EXPERT_WEIGHTS[idx] * weights
    elif method == "Confidence-Based":
        # Weighted by expert comparison confidence scores
        w = np.zeros(6)
        for idx, (name, weights) in enumerate(EXPERT_AHP_WEIGHTS.items()):
            w += CONF_EXPERT_WEIGHTS[idx] * weights
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
        
    # Ensure it is normalized to sum to 1.0
    return w / np.sum(w)

def generate_raw_criteria(n_athletes=30, seed=42):
    """
    Generates synthetic raw risk criteria scores for a set of athletes.
    Scores represent scaled values (0.0 to 1.0) representing risk severity.
    """
    np.random.seed(seed)
    
    first_names = [
        "Ahmet", "Mehmet", "Ali", "Mustafa", "Can", "Burak", "Emre", "Ömer", "Arda", "Yiğit",
        "Ayşe", "Fatma", "Zeynep", "Elif", "Yağmur", "Merve", "Aslı", "Selin", "Ecem", "Buse",
        "Deniz", "Ege", "Umut", "Görkem", "Barış", "Kaan", "Dilek", "Seda", "Gizem", "İrem"
    ]
    last_names = [
        "Yılmaz", "Kaya", "Demir", "Şahin", "Çelik", "Yıldız", "Erdoğan", "Öztürk", "Aydın", "Özdemir",
        "Arslan", "Doğan", "Kılıç", "Aslan", "Çetin", "Kara", "Koç", "Kurt", "Özkan", "Şen"
    ]
    
    names = []
    for i in range(n_athletes):
        fname = first_names[i % len(first_names)]
        lname = last_names[np.random.randint(0, len(last_names))]
        names.append(f"{fname} {lname}")
        
    data = {
        "Athlete_ID": [f"ATH_{i+1:03d}" for i in range(n_athletes)],
        "Name": names
    }
    
    # Generate random scores for the 6 biomechanical criteria (between 0.0 and 1.0)
    for criterion in CRITERIA:
        # Uniform distribution with slightly different ranges to mimic different variance
        if criterion == "Knee_Mechanics" or criterion == "Overall_Landing_Quality":
            # These are weighted highest by experts, so let's give them a wider range of risks
            data[criterion] = np.random.uniform(0.1, 0.95, n_athletes)
        else:
            data[criterion] = np.random.uniform(0.1, 0.85, n_athletes)
            
    return pd.DataFrame(data)

def compute_owa_score(df, owa_weights, ahp_weights):
    """
    Computes risk score using the combination of AHP and OWA.
    
    1. Weigh the criteria using the AHP weights to get AHP weighted scores.
    2. Sort the weighted scores descending.
    3. Apply the OWA weights to get aggregated risk score.
    4. Scale score to 0-10 to match clinical LESS score representation.
    """
    assert len(ahp_weights) == len(CRITERIA), "AHP weights must match number of criteria."
    assert len(owa_weights) == len(CRITERIA), "OWA weights must match number of criteria."
    
    # Weigh criteria by AHP weights
    weighted_criteria = df[CRITERIA].values * ahp_weights
    
    # Sort weighted scores descending for OWA aggregation
    sorted_weighted = np.sort(weighted_criteria, axis=1)[:, ::-1]
    
    # Apply OWA weights
    owa_scores = np.dot(sorted_weighted, owa_weights)
    
    # Normalize OWA score to maximum possible weighted value, then scale to 0-10.
    # Maximum possible score is achieved when all raw criteria are 1.0.
    # When raw criteria are 1.0, weighted criteria is simply ahp_weights.
    # Then sorting ahp_weights descending and applying owa_weights gives the max possible score.
    max_possible_weighted = np.dot(np.sort(ahp_weights)[::-1], owa_weights)
    
    scaled_scores = (owa_scores / max_possible_weighted) * 10.0
    
    result_df = df.copy()
    result_df["OWA_Risk_Score"] = scaled_scores
    return result_df

if __name__ == "__main__":
    df = generate_raw_criteria(n_athletes=5)
    ahp_w = get_aggregated_ahp_weights("Neutral / Average")
    
    # OWA weights: prioritize the worst scores (pessimistic / conservative risk aggregation)
    owa_w = np.array([0.35, 0.25, 0.20, 0.10, 0.07, 0.03])
    
    df_owa = compute_owa_score(df, owa_w, ahp_w)
    print("--- 6 Criteria Data Loader Test (Scores Scaled 0-10) ---")
    print(df_owa[["Athlete_ID", "Name", "OWA_Risk_Score"]])
