import streamlit as st
import pandas as pd
import pickle
import altair as alt

from config import (
    MODEL_PATH,
    CLEANED_TRAINING_DATA_PATH,
    ACTIVE_PLAYERS_PATH,
    FEATURES,
    CLASS_LABELS,
)
from utils import calculate_expected_impact, generate_clinical_narrative


WARNING_THRESHOLD = 0.15
CAUTION_THRESHOLD = 0.08


st.set_page_config(page_title="Risk Stratification Dashboard", layout="wide")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_player_state():
    df = pd.read_csv(CLEANED_TRAINING_DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df


@st.cache_data
def load_active_players():
    active = pd.read_csv(ACTIVE_PLAYERS_PATH)
    return active["player_name"].dropna().unique().tolist()


def format_probability(prob):
    return f"{prob * 100:.2f}%"


def get_latest_player_row(df, player_name):
    player_rows = df[df["player_name"] == player_name].sort_values("GAME_DATE", ascending=False)
    if player_rows.empty:
        return None
    return player_rows.iloc[0]


def build_input_row(latest_row, projected_minutes, back_to_back):
    baseline = latest_row.copy()

    input_row = {
        "MIN": projected_minutes,
        "Back_to_Back": int(back_to_back),
    }

    for feature in FEATURES:
        if feature in input_row:
            continue
        input_row[feature] = baseline.get(feature, None)

    return pd.DataFrame([input_row])


def render_risk_snapshot(col, label, prob, status_color):
    color = status_color
    col.markdown(
        """
        <div style="text-align:left;">
            <div style="font-size:0.85rem;color:{color};">{label}</div>
            <div style="font-size:1.25rem;font-weight:600;color:{color};">{value}</div>
        </div>
        """.format(label=label, value=format_probability(prob), color=color),
        unsafe_allow_html=True,
    )


def get_injury_history(df, player_name):
    try:
        player_rows = df[df["player_name"] == player_name]
        if player_rows.empty:
            return "No recent injuries on record."
        recent = player_rows[player_rows["Injury_Class"] > 0]
        if recent.empty:
            return "No recent injuries on record."
        recent_dates = recent.sort_values("GAME_DATE", ascending=False).head(3)["GAME_DATE"]
        return " | ".join([d.strftime("%Y-%m-%d") for d in recent_dates if pd.notna(d)])
    except Exception:
        return "Injury history unavailable."


model = load_model()
player_state = load_player_state()
active_players = load_active_players()

player_state = player_state[player_state["player_name"].isin(active_players)]

st.markdown(
    "<h1>NBA Injury Track: <span style='font-size:0.65em;font-weight:400'>"
    "a Longitudal Risk & Triage Dashboard</span></h1>",
    unsafe_allow_html=True,
)
st.caption("This website is based on NBA data from 2016–2025.")


def render_player_section(
    container,
    section_key,
    section_title,
    player_name,
    projected_minutes,
    back_to_back,
    chart_color,
):
    latest_row = get_latest_player_row(player_state, player_name)
    if latest_row is None:
        container.error("No player data available for the selected athlete.")
        return

    input_df = build_input_row(latest_row, projected_minutes, back_to_back)
    raw_probabilities = model.predict_proba(input_df)[0]
    severity_order = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
    prob_map = {CLASS_LABELS[i]: raw_probabilities[i] for i in sorted(CLASS_LABELS.keys())}

    container.subheader(section_title)
    container.markdown(f"**{player_name}**")

    alert_text, impact_score = calculate_expected_impact([raw_probabilities])

    col1, col2, col3 = container.columns(3)
    with col1:
        st.metric("Projected Minutes", f"{projected_minutes} min")
    with col2:
        st.metric("Back-to-Back Game", "Yes" if back_to_back else "No")
    with col3:
        if "CRITICAL" in alert_text:
            st.error(alert_text)
        elif "WARNING" in alert_text:
            st.warning(alert_text)
        else:
            st.success(alert_text)

    injury_probs = [(label, prob) for label, prob in prob_map.items() if label != "Healthy"]
    injury_probs_sorted = sorted(injury_probs, key=lambda x: x[1], reverse=True)
    top_risks = injury_probs_sorted[:2]

    top_risk_text = ", ".join(
        [f"{label} ({format_probability(prob)})" for label, prob in top_risks]
    )

    container.markdown("---")
    container.subheader("Automated Medical Assessment (XAI)")

    with container.spinner("Synthesizing patient history and ML outputs via Groq..."):
        history_text = get_injury_history(player_state, player_name)
        narrative = generate_clinical_narrative(
            player_name,
            latest_row.get("Acute_Load_5G", projected_minutes),
            latest_row.get("Chronic_Load_15G", 0),
            [raw_probabilities],
            f"Top Risks: {top_risk_text} | Injury History: {history_text} | "
            f"Height: {latest_row.get('Height_Tier', 'Unknown')}, "
            f"Weight: {latest_row.get('Weight_Tier', 'Unknown')}, "
            f"BMI: {latest_row.get('BMI_Tier', 'Unknown')}, "
            f"Projected Minutes: {projected_minutes} | "
            f"{player_name} on {projected_minutes} minutes played | "
            f"Back-to-Back: {'Yes' if back_to_back else 'No'}",
        )
        container.info(narrative)
        container.caption(
            "Generated by the Llama 3.1 8B (Groq) model using workload context. "
            "Verify with NBA logs; AI isn't always right. Double check."
        )

    container.markdown("---")
    container.subheader("Outcome Probability Distribution")

    show_healthy_key = f"show_healthy_{section_key}"
    if show_healthy_key not in st.session_state:
        st.session_state[show_healthy_key] = True

    show_healthy = st.session_state[show_healthy_key]
    display_order = (
        severity_order
        if show_healthy
        else [label for label in severity_order if label != "Healthy"]
    )
    display_probs = [prob_map[label] for label in display_order]

    prob_df = pd.DataFrame(
        {
            "Outcome": display_order,
            "Probability": display_probs,
        }
    )

    if show_healthy:
        y_max = 1.0
    else:
        y_max = max(display_probs) if display_probs else 1.0
        y_max = max(y_max, 0.05)

    chart = (
        alt.Chart(prob_df)
        .mark_bar(color=chart_color)
        .encode(
            x=alt.X(
                "Outcome:N",
                sort=severity_order,
                title="Outcome",
                axis=alt.Axis(
                    labelAngle=0,
                    labelFontSize=10,
                ),
            ),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, y_max]), title="Probability"),
            tooltip=[
                alt.Tooltip("Outcome:N"),
                alt.Tooltip("Probability:Q", format=".2f"),
            ],
        )
    )
    container.altair_chart(chart, use_container_width=True)
    container.checkbox("Include Healthy class", key=show_healthy_key)
    container.caption("Probabilities shown are from the XGBoost injury model.")

    container.markdown("---")
    container.subheader("Clinical Risk Snapshot")

    cols = container.columns(3)
    for idx, label in enumerate(severity_order):
        prob = prob_map[label]
        col = cols[idx % 3]
        if label == "Healthy":
            status_color = "#16A34A"
        elif prob >= WARNING_THRESHOLD:
            status_color = "#DC2626"
        elif prob >= CAUTION_THRESHOLD:
            status_color = "#D97706"
        else:
            status_color = "#16A34A"
        render_risk_snapshot(col, label, prob, status_color)


controls = st.columns(2)

with controls[0]:
    st.subheader("Primary Player")
    selected_player = st.selectbox(
        "Search Player 🔍",
        active_players,
        help="Type to search players",
        key="primary_player",
        index=None,
        placeholder="Start typing a player name...",
    )
    if selected_player:
        primary_rows = player_state[player_state["player_name"] == selected_player]
        avg_minutes = float(primary_rows["MIN"].mean()) if not primary_rows.empty else 24
        slider_col, b2b_col = st.columns([4, 1])
        projected_minutes = slider_col.slider(
            "Projected Minutes",
            min_value=0,
            max_value=48,
            value=int(round(avg_minutes)) if not pd.isna(avg_minutes) else 24,
            key="primary_minutes",
        )
        back_to_back = b2b_col.checkbox("B2B", value=False, key="primary_b2b")
    else:
        projected_minutes = None
        back_to_back = False

with controls[1]:
    st.subheader("Comparison Player")
    comparison_options = (
        [p for p in active_players if p != selected_player]
        if selected_player
        else []
    )
    comparison_player = st.selectbox(
        "Compare Player 🔍",
        comparison_options,
        help="Optional comparison player",
        key="comparison_player",
        index=None,
        placeholder="Start typing a player name...",
    )
    comparison_minutes = None
    comparison_b2b = False
    if comparison_player:
        comparison_rows = player_state[player_state["player_name"] == comparison_player]
        comparison_avg = float(comparison_rows["MIN"].mean()) if not comparison_rows.empty else 24
        slider_col, b2b_col = st.columns([4, 1])
        comparison_minutes = slider_col.slider(
            "Projected Minutes",
            min_value=0,
            max_value=48,
            value=int(round(comparison_avg)) if not pd.isna(comparison_avg) else 24,
            key="comparison_minutes",
        )
        comparison_b2b = b2b_col.checkbox("B2B", value=False, key="comparison_b2b")

st.markdown("---")
footer = "Source code: https://github.com/waitthisisntsteam/NBAInjuryTrack/tree/main"

if not selected_player or projected_minutes is None:
    st.info("Search for a player above to begin.")
    st.caption(footer)
    st.stop()

if comparison_player and comparison_minutes is not None:
    output_cols = st.columns(2)
    render_player_section(
        output_cols[0],
        "primary",
        "Primary Analysis",
        selected_player,
        projected_minutes,
        back_to_back,
        "#2A6F9B",
    )
    render_player_section(
        output_cols[1],
        "comparison",
        "Comparison Analysis",
        comparison_player,
        comparison_minutes,
        comparison_b2b,
        "#D97706",
    )
else:
    render_player_section(
        st.container(),
        "primary",
        "Primary Analysis",
        selected_player,
        projected_minutes,
        back_to_back,
        "#2A6F9B",
    )

st.markdown("---")
st.caption("Source code: https://github.com/waitthisisntsteam/NBAInjuryTrack/tree/main")
