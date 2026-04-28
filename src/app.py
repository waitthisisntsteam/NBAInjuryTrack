import streamlit as st
import pandas as pd
import pickle
from config import (
    MODEL_PATH,
    CLEANED_TRAINING_DATA_PATH,
    CLEANED_INJURIES_PATH,
    ACTIVE_PLAYERS_PATH,
    FEATURES,
    FEATURE_WINDOW_ACUTE,
    FEATURE_WINDOW_CHRONIC,
    CLASS_LABELS
)
from utils import calculate_expected_impact, generate_clinical_narrative

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Risk Stratification Dashboard", layout="wide")

# --- 1. LOAD DATA & MODEL ---
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv(CLEANED_TRAINING_DATA_PATH)
    latest_stats = df.sort_values("GAME_DATE").groupby("player_name").last().reset_index()
    return latest_stats

@st.cache_data
def load_active_players():
    active = pd.read_csv(ACTIVE_PLAYERS_PATH)
    return active["player_name"].dropna().unique().tolist()

def get_injury_history(player_name):
    try:
        injuries = pd.read_csv(CLEANED_INJURIES_PATH)
        player_injuries = injuries[injuries["player_name"] == player_name]
        if player_injuries.empty:
            return "No recent injuries on record."

        recent_reasons = player_injuries["Reason"].tail(3).dropna().tolist()
        return " | ".join(recent_reasons)
    except Exception:
        return "Injury history unavailable."

model = load_model()
player_data = load_data()
active_players = load_active_players()
player_data = player_data[player_data["player_name"].isin(active_players)]

# --- 2. SIDEBAR: PATIENT INTAKE & SCENARIOS ---
st.sidebar.title("Patient Intake")
selected_player = st.sidebar.selectbox("Select Player Profile", player_data["player_name"].unique())

baseline = player_data[player_data['player_name'] == selected_player].iloc[0]

st.sidebar.markdown("---")
st.sidebar.subheader("Clinical 'What-If' Scenario")
st.sidebar.write("Adjust simulated minutes for tonight's game to see how acute workload shifts risk probabilities.")

acute_col = FEATURES[1]
chronic_col = FEATURES[2]

simulated_minutes = st.sidebar.slider(
    "Simulated Game Minutes",
    min_value=0, max_value=48, value=int(baseline[acute_col])
)

# Dynamically calculate the new acute load using the window constant
simulated_acute_load = ((baseline[acute_col] * (FEATURE_WINDOW_ACUTE - 1)) + simulated_minutes) / FEATURE_WINDOW_ACUTE

input_features = pd.DataFrame({
    'MIN': [simulated_minutes],
    acute_col: [simulated_acute_load],
    chronic_col: [baseline[chronic_col]]
})

# --- 3. RUN ML & TRIAGE MATH ---
raw_pred = model.predict_proba(input_features)[0]

# DEFENSIVE ENGINEERING: Pad the array if the sample model didn't learn all 3 classes
padded_probs = list(raw_pred)
while len(padded_probs) < 3:
    padded_probs.append(0.0)

# Wrap back in a 2D array so the rest of the app parses it correctly
raw_probabilities = [padded_probs]

alert_text, impact_score = calculate_expected_impact(raw_probabilities)

# --- 4. MAIN DASHBOARD UI ---
st.title("Longitudinal Risk & Triage Dashboard")
st.markdown(f"**Monitoring Profile:** `{selected_player}`")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(f"Acute Workload ({FEATURE_WINDOW_ACUTE}G Avg)", f"{simulated_acute_load:.1f} min",
              delta=f"{simulated_acute_load - baseline[acute_col]:.1f} min from baseline",
              delta_color="inverse")
with col2:
    st.metric(f"Chronic Load ({FEATURE_WINDOW_CHRONIC}G Avg)", f"{baseline[chronic_col]:.1f} min")
with col3:
    if "CRITICAL" in alert_text:
        st.error(f"🚨 {alert_text}")
    elif "WARNING" in alert_text:
        st.warning(f"⚠️ {alert_text}")
    else:
        st.success(f"✅ {alert_text}")

st.markdown("---")
st.subheader("Outcome Probability Distribution")

prob_df = pd.DataFrame({
    "Outcome Classification": [
        f"{CLASS_LABELS[0]} (0)",
        f"{CLASS_LABELS[1]} (1)",
        f"{CLASS_LABELS[2]} (2)"
    ],
    "Probability (%)": [raw_probabilities[0][0]*100, raw_probabilities[0][1]*100, raw_probabilities[0][2]*100]
}).set_index("Outcome Classification")

st.bar_chart(prob_df, use_container_width=True)

st.markdown("---")
st.subheader("Automated Medical Assessment (XAI)")

if st.button("Generate Explainable AI Narrative"):
    with st.spinner("Synthesizing patient history and ML outputs via Groq..."):
        history_text = get_injury_history(selected_player)
        narrative = generate_clinical_narrative(
            selected_player,
            simulated_acute_load,
            baseline[chronic_col],
            raw_probabilities,
            history_text
        )
        st.info(narrative)
else:
    st.write("*Click the button above to run the LLM synthesis based on the current slider parameters.*")
