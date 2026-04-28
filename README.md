# NBA Injury Track: a Longitudal Risk & Triage Dashboard

## Overview
NBA Injury Track is a full-stack, end-to-end pipeline that turns NBA game logs, biometrics, and public injury reports into a clinical-style risk stratification dashboard. It uses engineered workload and biometric features to generate multi-class injury probabilities with XGBoost, then synthesizes an explanatory narrative using a Groq-hosted Llama model.

The system is built around three outputs:
- **Risk probabilities:** six-class injury classification from XGBoost.
- **Triage signal:** a severity-weighted expected impact alert.
- **Clinical narrative:** a short XAI summary grounded in workload, biometrics, and recent injury context.

## Data Sources
We use static Kaggle CSVs:
- `PlayerStatisticsExtended.csv`
- `Players.csv`
- `injury_data.csv`

All data is constrained to **2016–2025** to match the injury dataset coverage. (I got IP banned on my university wifi and VPN)

## Pipeline Summary
1. **Prep Kaggle data**
   - Clean game logs, biometrics, and injuries.
   - Normalize names and filter noise (rest/illness/management).
   - Build a list of active players (2024–2025) for the UI.

2. **Feature engineering**
   - Expanded workload windows and rolling loads.
   - Acute/Chronic ratio and short-horizon minute sums.
   - BMI + biometric tiers + position group.
   - Usage/Pace mix and back-to-back flags.
   - 6-tier injury classification mapped from text.

3. **Model training**
   - XGBoost with class-weighted loss for severe imbalance.
   - Outputs multi-class probabilities.

4. **Streamlit UI**
   - Active player search + optional comparison player.
   - Projection inputs (minutes + back-to-back).
   - Probability distribution + clinical narrative.

## Injury Classes
The model predicts a 6-tier orthopedic severity scale:
0. Healthy
1. Fatigue/Soreness
2. Contusion/Impact
3. Joint Sprain
4. Muscular Strain
5. Structural/Surgical

## Project Structure
```
NBAInjuryTrack/
├─ data/
│  ├─ original/        # raw Kaggle CSVs
│  ├─ prep_data/       # cleaned injuries, merged logs, active players
│  └─ training/        # cleaned_training_data.csv
├─ models/             # saved XGBoost model
├─ src/
│  ├─ prep_kaggle_data.py
│  ├─ feature_engineering.py
│  ├─ model_training.py
│  ├─ app.py           # streamlit website builder
│  └─ utils.py         # streamlit website helper
└─ requirements.txt
```

## Setup
[Players & PlayerStatisticsExtended](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores?select=Games.csv)
[injury_data](https://www.kaggle.com/datasets/jacquesoberweis/2016-2025-nba-injury-data)


1. Place Kaggle CSVs:
```
data/original/Players.csv
data/original/PlayerStatisticsExtended.csv
data/original/injury_data.csv
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the pipeline:
```
python src/prep_kaggle_data.py
python src/feature_engineering.py
python src/model_training.py
```

4. Launch Streamlit:
```
streamlit run src/app.py
```

## Notes
- The model is trained on historical injury outcomes and predicts probabilities, not diagnoses.
- The LLM narrative is explanatory only and can be incomplete or incorrect.

## Credits
- Kaggle datasets listed above
- NBA public injury reports
- Groq API for LLM synthesis
