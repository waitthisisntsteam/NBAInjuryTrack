import os
from groq import Groq
from config import SEVERITY_MULTIPLIERS, CLASS_LABELS

# --- 1. THE CLINICAL TRIAGE ENGINE (Expected Impact Math) ---
def calculate_expected_impact(probabilities):
    """
    Translates raw ML probabilities into a Severity-Weighted Risk Alert.
    Expects a list/array aligned with CLASS_LABELS.
    """
    probs = probabilities[0]

    impacts = {
        class_idx: probs[class_idx] * SEVERITY_MULTIPLIERS[class_idx]
        for class_idx in SEVERITY_MULTIPLIERS
        if class_idx != 0 and class_idx < len(probs)
    }

    if not impacts:
        return "OPTIMAL: Baseline Risk Level", 0.0

    max_class = max(impacts, key=impacts.get)
    max_impact = impacts[max_class]

    if max_impact >= 1.0:
        return "CRITICAL: High Injury Risk", max_impact
    if max_impact >= 0.5:
        return "WARNING: Elevated Injury Risk", max_impact
    return "OPTIMAL: Baseline Risk Level", max_impact

# --- 2. THE EXPLAINABLE AI SYNTHESIS (Groq API) ---
def generate_clinical_narrative(player_name, acute_load, chronic_load, ml_probs, injury_history_text):
    """
    Uses the Groq API to synthesize a 1-2 sentence risk analysis.
    """
    try:
        client = Groq()
        probs = ml_probs[0]
        risk_lines = ", ".join(
            [
                f"{CLASS_LABELS[i]} {probs[i]*100:.1f}%"
                for i in sorted(CLASS_LABELS.keys())
                if i < len(probs)
            ]
        )

        prompt = f"""
        Player: {player_name}
        Workload Vitals: {acute_load:.1f} mins/game (Recent 5G), {chronic_load:.1f} mins/game (Season 15G).
        XGBoost Risk Probabilities: {risk_lines}
        Past Injury Context: {injury_history_text}
        """

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a sports risk analyst. I will provide an injury prediction, recent workload, and past injury history. Write a simple, 1 to 2 sentence summary explaining the player's current risk level. Keep the explanation concise. You must strictly avoid giving any medical treatment, rehabilitation, or recovery advice."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=150
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"System Note: LLM synthesis unavailable. Please verify API key. (Error: {str(e)})"
