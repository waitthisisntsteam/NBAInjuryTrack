import os
from groq import Groq
from config import SEVERITY_MULTIPLIERS

# --- 1. THE CLINICAL TRIAGE ENGINE (Expected Impact Math) ---
def calculate_expected_impact(probabilities):
    """
    Translates raw ML probabilities into a Severity-Weighted Risk Alert.
    Expects a list/array like: [p_healthy, p_soft_tissue, p_severe]
    """
    p_healthy, p_soft, p_severe = probabilities[0]

    # Calculate expected impact using the explicitly imported dictionary
    impact_soft = p_soft * SEVERITY_MULTIPLIERS[1]
    impact_severe = p_severe * SEVERITY_MULTIPLIERS[2]

    # Determine the highest threat based on impact, not just raw probability
    if impact_severe > impact_soft and impact_severe > 0.5:
        return "CRITICAL: High Structural Risk", impact_severe
    elif impact_soft > impact_severe and impact_soft > 0.5:
        return "WARNING: Elevated Soft Tissue Risk", impact_soft
    else:
        return "OPTIMAL: Baseline Risk Level", 0.0

# --- 2. THE EXPLAINABLE AI SYNTHESIS (Groq API) ---
def generate_clinical_narrative(player_name, acute_load, chronic_load, ml_probs, injury_history_text):
    """
    Uses the Groq API to synthesize a 1-2 sentence risk analysis.
    """
    try:
        client = Groq()

        p_healthy, p_soft, p_severe = ml_probs[0]

        prompt = f"""
        Player: {player_name}
        Workload Vitals: {acute_load:.1f} mins/game (Recent 5G), {chronic_load:.1f} mins/game (Season 15G).
        XGBoost Risk Probabilities: {p_healthy*100:.1f}% Healthy, {p_soft*100:.1f}% Soft Tissue, {p_severe*100:.1f}% Severe.
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
