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
    predict_score.py       championship game total score predictor
    predict_ot.py          Kristin's Challenge — OT game predictor
  predictions/
    bracket_2026.csv
    championship_odds_2026.csv
    feature_importance_2026.png
    kristins_challenge_2026.json
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
- **KenPom Preseason** (`KenPom Preseason.csv`): preseason AdjEM and rank per team per year, used to compute trajectory (how much a team improved from preseason to tournament)
- **AP Poll** (`AP Poll.csv`): weekly AP top-25 rankings per team per year, used to compute final pre-tournament AP rank as a momentum signal
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
- **Upset and toss-up signals** (added for improved close-game accuracy):
  - `abs_em_gap` / `is_tossup`: absolute efficiency margin gap and a binary flag for games within 5 points (true toss-ups)
  - `delta_ps_em_change` / `delta_ps_rank_change`: difference in KenPom preseason trajectory between teams — peaking teams tend to outperform expectations
  - `delta_peaking`: binary momentum flag for teams that improved 10+ spots from preseason to tournament
  - `delta_ap_rank_final` / `delta_ranked`: final AP rank differential and whether each team was ranked at all — captures national recognition and momentum
  - `delta_3pt_volatility`: three-point attempt rate x (1 - three-point percentage) — high volatility teams swing more in unpredictable games
  - `pace_control_edge`: which team is better equipped to control game tempo — slow-paced teams can neutralize faster, higher-efficiency opponents
  - `delta_ft_clutch`: free throw percentage differential — better FT shooting matters most in close late-game situations
  - `delta_recent_hist`: weighted combination of historical tournament win rate and Sweet 16 rate — program experience in high-pressure moments

Recursive Feature Elimination with 5-fold cross-validation identifies the optimal subset before final model training. The current selected feature set contains 33 features (up from the original 24).

With the expanded feature set, the model shows meaningful improvement in the areas that matter most for bracket accuracy: upset recall improved from 55% to 58.3%, close-game accuracy (games within 10 efficiency margin) improved from 63% to 70%, and overall Brier score improved from 0.1390 to 0.1358. The new features help specifically in toss-up matchups without affecting the model's confidence on clear favorites.

## Models

| Model | CV Log-Loss | 2019 Accuracy | 2022 Accuracy | 2023 Accuracy |
|-------|------------|--------------|--------------|--------------|
| Logistic Regression | ~0.55 | 79.4% | 82.5% | 74.6% |
| Ridge (calibrated) | ~0.55 | 79.4% | 79.4% | 74.6% |
| Random Forest | ~0.60 | 73.8% | 71.4% | 70.6% |
| LightGBM | ~0.58 | 75.4% | 76.2% | 79.4% |
| Ensemble (Ridge + RF) | ~0.57 | 79.4% | 74.6% | 74.6% |

Models are trained on years 2008-2023 excluding the held-out evaluation years 2019, 2022, and 2023.

## Championship Score Prediction

`src/predict_score.py` predicts the total combined points in the championship game using a tempo-adjusted efficiency formula ensembled across both Barttorvik and KenPom rating systems.

```
TeamA_pts = (TeamA_AdjO / 100) x (TeamB_AdjD / 100) x possessions
TeamB_pts = (TeamB_AdjO / 100) x (TeamA_AdjD / 100) x possessions
possessions = average tempo of both teams
```

A 4.5% tournament adjustment is applied to account for elevated defensive intensity in late-round games. For 2026, the predicted championship total is **155 points** (Florida vs Arizona).

```bash
python src/predict_score.py Florida Arizona 2026
```

## Kristin's Challenge — OT Game Prediction

`src/predict_ot.py` predicts the number of overtime games in the Round of 64 (Thursday + Friday games) for a given tournament year.

**Methodology:** A logistic regression model is trained on 17 years of historical Round of 64 games using the following features per matchup:

- `abs_em_diff`: absolute efficiency margin gap between the two teams — tighter matchups are more likely to go to OT
- `avg_tempo`: average pace of both teams — slower-paced games have less scoring variance
- `seed_diff_abs`: absolute seed difference — evenly seeded games are more likely to be close
- `ft_pct_avg`: average free throw percentage — better FT shooting is critical in OT situations
- `ps_change_avg`: average KenPom preseason trajectory — both teams peaking into the tournament creates more tightly contested games
- `both_ranked`: whether both teams were in the AP top 25 — ranked matchups tend to be more competitive
- `avg_3pt_volatility`: three-point rate x miss rate — high-variance shooting teams produce more chaotic late-game situations

The model uses isotonic calibration (CalibratedClassifierCV) to anchor predicted probabilities to the historical OT base rate of 4.79%. Summing P(OT) across all 32 games gives the expected number of OT games.

**2026 Prediction:**
- Expected OT games: **1.25**
- Predicted OT games: **1**
- Historical average: ~1.4 OT games per Round of 64

Top OT candidates: Utah St. vs Villanova (6.7%), Missouri vs Miami FL (6.7%), Santa Clara vs Kentucky (5.8%)

```bash
python src/predict_ot.py 2026
```

Results saved to `predictions/kristins_challenge_2026.json`.
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

Predict championship game total:
```bash
python src/predict_score.py Florida Arizona 2026
```

Predict Round of 64 OT games (Kristin's Challenge):
```bash
python src/predict_ot.py 2026
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
  Predicted total: 155 points
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
- `predictions/kristins_challenge_2026.json` — per-game OT probabilities and expected OT count for 2026 Round of 64
