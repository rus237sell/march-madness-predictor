# march-madness-predictor

A Python-based machine learning pipeline for predicting NCAA Men's Basketball Tournament game winners. The system trains on historical tournament data from 2008 through 2023, uses an ensemble of Ridge Regression, Logistic Regression, Random Forest, and LightGBM models, and runs 10,000 Monte Carlo simulations to generate championship probability distributions rather than a single deterministic bracket.

## Repository Structure

```
march-madness-predictor/
  data/
    raw/           raw CSV data sources
    processed/     merged and cleaned datasets
  notebooks/
    01_eda.ipynb
    02_feature_engineering.ipynb
    03_model_training.ipynb
    04_bracket_simulation.ipynb
  src/
    data_loader.py         load and merge all data sources
    feature_engineering.py compute matchup-level features and run RFE
    model.py               train and evaluate all models
    simulate_bracket.py    run Monte Carlo bracket simulation
  predictions/
    bracket_2026.csv
    championship_odds_2026.csv
    feature_importance_2026.png
  requirements.txt
  README.md
```

## Data Sources

- **KenPom / Barttorvik combined CSV** (`KenPom Barttorvik.csv`): AdjO, AdjD, AdjEM, AdjT, Barthag, Four Factors, shooting splits, experience, talent, SOS, PPP offensive and defensive, 2008-2026
- **Tournament Matchups** (`Tournament Matchups.csv`): game-level tournament results by year and round, used to build the labeled training dataset
- **Resumes** (`Resumes.csv`): Q1 wins, Q2 wins, NET RPI, ELO, resume score per team per year
- **Shooting Splits** (`Shooting Splits.csv`): dunk rate, close twos, mid-range, three-point share and percentage splits
- **Barttorvik Away-Neutral** (`Barttorvik Away-Neutral.csv`): neutral-site adjusted efficiency metrics
- **EvanMiya ratings** (`EvanMiya.csv`): O Rate, D Rate, Relative Rating, True Tempo, Kill Shots
- **Z Ratings** (`Z Rating Teams.csv`): composite Z-rating per team per year
- **Teamsheet Ranks** (`Teamsheet Ranks.csv`): detailed quad win/loss breakdown per team
- **Conference Stats** (`Conference Stats.csv`): conference-level AdjEM for SOS context
- **Team Results** (`Team Results.csv`): historical program tournament win rate, S16/F4/championship rates
- **Coach Results** (`Coach Results.csv`): career tournament win rate, Final Four appearances per coach
- **Tournament Simulation** (`Tournament Simulation.csv`): current year bracket structure for simulation

## Feature Engineering

Features are computed at the matchup level. Continuous stats are expressed as the delta between Team A and Team B unless the feature is team-specific. Key feature categories:

- Efficiency: AdjEM delta, Barthag delta, efficiency ratio (AdjO/AdjD) per team and delta, neutral-site AdjEM delta
- Four Factors: eFG%, turnover rate, offensive/defensive rebound rate, free throw rate deltas
- Shooting profile: three-point rate and percentage, two-point percentage, mid-range rate
- Tempo and style: AdjT delta, absolute tempo mismatch, pace classification (slow/medium/fast)
- Resume: Q1 wins delta, ELO delta, resume score delta, elite SOS delta, WAB delta
- Program history: tournament win rate, S16 rate, F4 rate, blue blood indicator
- Composite: balanced profile score, trapezoid of excellence flag, tournament readiness index
- Seed features: seed difference, seed ratio

Recursive Feature Elimination with 5-fold cross-validation identifies the optimal subset before final model training.

## Models

| Model | CV Log-Loss | 2019 Accuracy | 2022 Accuracy | 2023 Accuracy |
|-------|------------|--------------|--------------|--------------|
| Logistic Regression | ~0.55 | 79.4% | 82.5% | 74.6% |
| Ridge (calibrated) | ~0.55 | 79.4% | 79.4% | 74.6% |
| Random Forest | ~0.60 | 73.8% | 71.4% | 70.6% |
| LightGBM | ~0.58 | 75.4% | 76.2% | 79.4% |
| Ensemble (Ridge + RF) | ~0.57 | 79.4% | 74.6% | 74.6% |

Models are trained on years 2008-2023 excluding the held-out evaluation years 2019, 2022, and 2023.

## Running the Pipeline

Install dependencies:
```bash
pip install -r requirements.txt
```

Build data and train models:
```bash
python src/data_loader.py
python src/model.py
```

Run bracket simulation for the current year:
```bash
python src/simulate_bracket.py --year 2026 --sims 10000
```

Or use the notebooks in order:
```
notebooks/01_eda.ipynb
notebooks/02_feature_engineering.ipynb
notebooks/03_model_training.ipynb
notebooks/04_bracket_simulation.ipynb
```

## 2026 Predicted Bracket

The bracket below shows the most-likely predicted path to the champion based on the Ridge model. Win probabilities shown are the probability that the left-side team wins the matchup.

```
ROUND OF 64
  (1) Duke          87%  -->  Duke
     (16) Siena
  (8) Ohio St.      90%  -->  Ohio St.
     (9) TCU
  (5) St. John's    50%  -->  St. John's
     (12) Northern Iowa
  (4) Kansas        62%  -->  Kansas
     (13) Cal Baptist
  (6) Louisville    83%  -->  Louisville
     (11) South Florida
  (3) Michigan St.  91%  -->  Michigan St.
     (14) North Dakota St.
  (7) UCLA          74%  -->  UCLA
     (10) UCF
  (2) Connecticut   88%  -->  Connecticut
     (15) Furman
  (1) Florida       97%  -->  Florida
     (16) Lehigh
  (8) Clemson       63%  -->  Clemson
     (9) Iowa
  (5) Vanderbilt    42%  -->  McNeese St.
     (12) McNeese St.
  (4) Nebraska      32%  -->  Troy
     (13) Troy
  (6) North Carolina 90% -->  North Carolina
     (11) VCU
  (3) Illinois      98%  -->  Illinois
     (14) Penn
  (7) Saint Mary's  41%  -->  Texas A&M
     (10) Texas A&M
  (2) Houston       75%  -->  Houston
     (15) Idaho
  (1) Arizona       97%  -->  Arizona
     (16) LIU Brooklyn
  (8) Villanova     92%  -->  Villanova
     (9) Utah St.
  (5) Wisconsin     95%  -->  Wisconsin
     (12) High Point
  (4) Arkansas      63%  -->  Arkansas
     (13) Hawaii
  (6) BYU           25%  -->  North Carolina St.
     (11) North Carolina St.
  (3) Gonzaga       98%  -->  Gonzaga
     (14) Kennesaw St.
  (7) Miami FL      76%  -->  Miami FL
     (10) Missouri
  (2) Purdue        51%  -->  Purdue
     (15) Queens
  (1) Michigan      99%  -->  Michigan
     (16) Howard
  (8) Georgia       16%  -->  Saint Louis
     (9) Saint Louis
  (5) Texas Tech    98%  -->  Texas Tech
     (12) Akron
  (4) Alabama       61%  -->  Alabama
     (13) Hofstra
  (6) Tennessee     97%  -->  Tennessee
     (11) SMU

ROUND OF 32
  (1) Duke         86%  -->  Duke
     (8) Ohio St.
  (5) St. John's   16%  -->  Kansas
     (4) Kansas
  (6) Louisville   63%  -->  Louisville
     (3) Michigan St.
  (7) UCLA         28%  -->  Connecticut
     (2) Connecticut
  (1) Florida      96%  -->  Florida
     (8) Clemson
  (12) McNeese St. 87%  -->  McNeese St.
     (13) Troy
  (6) North Carolina 46% --> Illinois
     (3) Illinois
  (10) Texas A&M    7%  -->  Houston
     (2) Houston
  (1) Arizona      82%  -->  Arizona
     (8) Villanova
  (5) Wisconsin    56%  -->  Wisconsin
     (4) Arkansas
  (11) NC State    18%  -->  Gonzaga
     (3) Gonzaga
  (7) Miami FL     37%  -->  Purdue
     (2) Purdue
  (1) Michigan     97%  -->  Michigan
     (9) Saint Louis
  (5) Texas Tech   61%  -->  Texas Tech
     (4) Alabama

SWEET 16
  (1) Duke         78%  -->  Duke
     (4) Kansas
  (6) Louisville   40%  -->  Connecticut
     (2) Connecticut
  (1) Florida      97%  -->  Florida
     (12) McNeese St.
  (3) Illinois     33%  -->  Houston
     (2) Houston
  (1) Arizona      74%  -->  Arizona
     (5) Wisconsin
  (3) Gonzaga      67%  -->  Gonzaga
     (2) Purdue
  (1) Michigan     79%  -->  Michigan
     (5) Texas Tech

ELITE EIGHT
  (1) Duke         64%  -->  Duke
     (2) Connecticut
  (1) Florida      66%  -->  Florida
     (2) Houston
  (1) Arizona      61%  -->  Arizona
     (3) Gonzaga

FINAL FOUR
  (1) Duke         41%  -->  FLORIDA
     (1) Florida

CHAMPIONSHIP
  Florida  (47.9% Monte Carlo win probability)
```

## 2026 Championship Odds (10,000 Monte Carlo Simulations)

| Team | Seed | Championship Probability |
|------|------|--------------------------|
| Florida | 1 | 47.9% |
| Duke | 1 | 21.9% |
| Houston | 2 | 10.8% |
| Connecticut | 2 | 8.1% |
| Louisville | 6 | 2.9% |
| Illinois | 3 | 2.4% |
| Kansas | 4 | 2.1% |
| Michigan St. | 3 | 0.9% |
| North Carolina | 6 | 0.9% |
| Ohio St. | 8 | 0.7% |

## Output Files

- `predictions/bracket_2026.csv` — deterministic bracket with round, teams, win probability, and predicted winner for each game
- `predictions/bracket_upset_adjusted_2026.csv` — alternate bracket resolving toss-up games (within 15%) toward the lower seed
- `predictions/championship_odds_2026.csv` — each team's championship probability from Monte Carlo simulations
- `predictions/feature_importance_2026.png` — Random Forest feature importance chart
- `predictions/model_eval_summary.json` — cross-validation and held-out year evaluation metrics per model
