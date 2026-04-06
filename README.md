
# NBA Injury Tracker & Predictor: A Longitudinal Risk Stratification & NLP Triage Dashboard

## Abstract
This project is an end-to-end Explainable AI (XAI) pipeline designed to simulate a **Clinical Decision Support System (CDSS)**. 

The system demonstrates how tracking real-time physical load and historical vulnerability can forecast acute health events, triage risks by severity, and provide physician-facing explanations using Generative AI. This architecture utilizes NBA player workloads, game logs, and public injury reports.

This project solves both issues by translating sports analytics into clinical informatics:
* **The Patient Profile:** NBA Injury History acts as a proxy for pre-existing conditions and longitudinal phenotyping.
* **The Vitals:** Game logs (minutes played, back-to-backs, usage rate) act as a proxy for acute physical stress and high-frequency biometric data.
* **The Output:** A triaged, interpretable dashboard designed to aid—not replace—human decision-making in high-stakes environments.

## System Architecture 

The application is built entirely in Python and operates through three distinct microservices:

### 1. The Machine Learning Engine (Risk Probability)
* **Model:** `XGBClassifier`.
* **Function:** Ingests engineered features (acute workload, chronic load, injury density) to output a multi-class probability distribution. 
* **Outcome:** Predicts the likelihood of specific event categories (e.g., Soft Tissue Strain vs. Ligamentous Injury vs. Healthy).

### 2. The Clinical Triage Engine (Expected Impact Score)
* **Function:** Raw probability is insufficient for clinical triage. A 20% risk of a catastrophic event requires more urgent attention than an 80% risk of a minor event.
* **Mechanism:** The system computes an **Expected Impact Score** by multiplying the ML probability by a static Severity Multiplier (e.g., ACL Tear = 250 days; Minor Sprain = 10 days). This mathematical layer automatically flags the most critical threat for the end-user.

### 3. The Explainable AI (XAI) Synthesis
* **Mechanism:** The system feeds the XGBoost numerical outputs, the recent workload stats, and the text-based injury history into a Large Language Model. The LLM is strictly prompted to synthesize this specific data into a concise, evidence-based "Clinical Narrative" to explain the risk factors to the practitioner.

## Tech Stack
* **Data Ingestion:** `nba_api`, `nbainjuries`
* **Data Engineering & Feature Extraction:** `pandas`, `numpy`
* **Predictive Modeling:** `xgboost`, `scikit-learn`
* **Generative AI / NLP:** `gemma4`
* - "You are a sports risk analyst. I will provide an injury prediction, recent workload, and past injury history. Write a simple, 1 to 2 sentence summary explaining the player's current risk level of ____ given by XGBoost. Keep the explanation concise. You must strictly avoid giving any medical treatment, rehabilitation, or recovery advice."
* **Frontend Framework:** `streamlit`

## Suggested UI/UX Implementation
* **Intake Sidebar:** Controls for selecting an individual profile.
* **Vitals & Alerts (Top Row):** High-level indicators displaying current fatigue parameters and the highest-severity risk alert (driven by the Expected Impact Score).
* **Probability Visualizations (Middle):** Horizontal bar charts plotting the full probability distribution of all potential outcomes.
* **Narrative Synthesis (Bottom):** A styled text container displaying the LLM-generated clinical summary, grounding the mathematical predictions in a readable, physician-friendly context.
