"""
helper script to generate all four Jupyter notebooks programmatically.
run with: python create_notebooks.py
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

NOTEBOOKS_DIR = os.path.dirname(os.path.abspath(__file__))


def make_notebook(cells):
    nb = new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def save_nb(nb, name):
    path = os.path.join(NOTEBOOKS_DIR, name)
    with open(path, "w") as f:
        nbformat.write(nb, f)
    print(f"Saved {path}")


# notebook 01: EDA
eda_cells = [
    new_markdown_cell("# 01 - Exploratory Data Analysis\n\nExplore the raw data sources: KenPom/Barttorvik stats, tournament matchups, and supplemental files."),
    new_code_cell("""
import sys, os
sys.path.insert(0, os.path.join('..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_loader import (
    load_kenpom_barttorvik, load_tournament_matchups,
    load_resumes, load_shooting_splits
)
"""),
    new_code_cell("""
kb = load_kenpom_barttorvik()
print(f"KenPom/Barttorvik shape: {kb.shape}")
print(f"Years: {kb['YEAR'].min()} - {kb['YEAR'].max()}")
kb.head(3)
"""),
    new_code_cell("""
# distribution of KADJ EM across tournament teams only
tourney = kb[kb['SEED'].notna() & (kb['SEED'] > 0)]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(tourney['KADJ EM'].dropna(), bins=30, edgecolor='black')
axes[0].set_title('KenPom AdjEM - Tournament Teams')
axes[0].set_xlabel('AdjEM')

axes[1].scatter(tourney['KADJ EM'], tourney['KADJ T'], alpha=0.3, s=10)
axes[1].set_xlabel('AdjEM')
axes[1].set_ylabel('AdjT (Tempo)')
axes[1].set_title('AdjEM vs Tempo')

axes[2].hist(tourney['SEED'].dropna(), bins=16, edgecolor='black')
axes[2].set_title('Seed Distribution')
axes[2].set_xlabel('Seed')

plt.tight_layout()
plt.show()
"""),
    new_code_cell("""
tm = load_tournament_matchups()
print(f"Tournament matchups shape: {tm.shape}")
print(f"Years: {tm['YEAR'].min()} - {tm['YEAR'].max()}")
print(f"Rounds: {sorted(tm['ROUND'].unique())}")

# win rate by seed
winners = tm[tm['ROUND'] < tm['CURRENT ROUND']].copy()
losers = tm[tm['ROUND'] == tm['CURRENT ROUND']].copy()
print(f"\\nWinners: {len(winners)}, Losers: {len(losers)}")
"""),
    new_code_cell("""
# upset frequency by seed matchup
import itertools

seed_wins = {}
for _, row in tm.iterrows():
    seed = int(row['SEED'])
    won = int(row['ROUND']) < int(row['CURRENT ROUND'])
    seed_wins.setdefault(seed, []).append(int(won))

seed_winrate = {s: (sum(v) / len(v)) for s, v in seed_wins.items() if len(v) >= 5}
seeds = sorted(seed_winrate.keys())
winrates = [seed_winrate[s] for s in seeds]

plt.figure(figsize=(10, 4))
plt.bar(seeds, winrates)
plt.axhline(0.5, color='red', linestyle='--', label='50%')
plt.xlabel('Seed')
plt.ylabel('Win Rate')
plt.title('Tournament Win Rate by Seed (all rounds, 2008-present)')
plt.legend()
plt.tight_layout()
plt.show()
"""),
    new_code_cell("""
# correlation heatmap of key efficiency metrics
import matplotlib.pyplot as plt
import numpy as np

key_cols = ['KADJ EM', 'KADJ O', 'KADJ D', 'KADJ T', 'BARTHAG',
            'EFG%', 'EFG%D', 'TOV%', 'OREB%', 'FT%', 'PPPO', 'PPPD']
available = [c for c in key_cols if c in kb.columns]
corr = kb[available].corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap='RdBu', vmin=-1, vmax=1)
ax.set_xticks(range(len(available)))
ax.set_yticks(range(len(available)))
ax.set_xticklabels(available, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(available, fontsize=8)
plt.colorbar(im, ax=ax)
ax.set_title('Correlation Heatmap - Key Metrics')
plt.tight_layout()
plt.show()
"""),
    new_code_cell("""
# show average KenPom AdjEM by round reached
round_adjem = kb[kb['SEED'].notna() & (kb['SEED'] > 0)].groupby('ROUND')['KADJ EM'].mean().sort_index()
round_labels = {64: 'R64', 32: 'R32', 16: 'S16', 8: 'E8', 4: 'F4', 2: 'Final', 1: 'Champ'}
round_adjem.index = [round_labels.get(r, str(r)) for r in round_adjem.index]

plt.figure(figsize=(8, 4))
plt.bar(round_adjem.index, round_adjem.values)
plt.xlabel('Deepest Round Reached')
plt.ylabel('Avg KenPom AdjEM')
plt.title('Average AdjEM by Deepest Tournament Round (2008-present)')
plt.tight_layout()
plt.show()
"""),
]

save_nb(make_notebook(eda_cells), "01_eda.ipynb")


# notebook 02: feature engineering
fe_cells = [
    new_markdown_cell("# 02 - Feature Engineering\n\nBuild the matchup-level feature matrix and inspect feature distributions."),
    new_code_cell("""
import sys, os
sys.path.insert(0, os.path.join('..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_loader import build_matchup_dataset, build_team_stats
from feature_engineering import prepare_training_data, impute_features, run_rfe
"""),
    new_code_cell("""
print("Building team stats (merging all data sources)...")
team_stats = build_team_stats()
print(f"Team stats shape: {team_stats.shape}")
print(f"Columns: {list(team_stats.columns[:20])}...")
"""),
    new_code_cell("""
print("Building matchup dataset...")
matchup_df = build_matchup_dataset(team_stats)
print(f"Matchup dataset shape: {matchup_df.shape}")
print(f"Years available: {sorted(matchup_df['YEAR'].unique())}")
print(f"\\nTarget distribution:")
print(matchup_df['TEAM_A_WIN'].value_counts())
"""),
    new_code_cell("""
# build feature matrix
X, y = prepare_training_data(matchup_df)
X = impute_features(X)
print(f"Feature matrix: {X.shape}")
print(f"\\nFeature list ({len(X.columns)} total):")
for i, c in enumerate(X.columns):
    print(f"  {i+1:3d}. {c}")
"""),
    new_code_cell("""
# distribution of the primary predictor: AdjEM delta
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

col = 'delta_badj_em'
if col in X.columns:
    wins = X[col][y == 1]
    losses = X[col][y == 0]
    axes[0].hist(wins, bins=40, alpha=0.6, label='Winner (Team A)', color='steelblue', edgecolor='none')
    axes[0].hist(losses, bins=40, alpha=0.6, label='Loser (Team A)', color='salmon', edgecolor='none')
    axes[0].axvline(0, color='black', linestyle='--')
    axes[0].set_xlabel('Barttorvik AdjEM Delta (A - B)')
    axes[0].set_title('AdjEM Delta: Winner vs Loser')
    axes[0].legend()

axes[1].scatter(X['delta_badj_em'], X['seed_diff'], c=y, cmap='coolwarm', alpha=0.2, s=5)
axes[1].set_xlabel('AdjEM Delta')
axes[1].set_ylabel('Seed Difference (A - B)')
axes[1].set_title('AdjEM Delta vs Seed Difference')

plt.tight_layout()
plt.show()
"""),
    new_code_cell("""
# check missing values
missing = X.isna().sum().sort_values(ascending=False)
print("Missing values per feature (top 20):")
print(missing.head(20))
"""),
    new_code_cell("""
# run RFE to find optimal feature subset (on training years only)
years = matchup_df['YEAR'].reset_index(drop=True)
train_mask = years <= 2023

print("Running RFECV with 5-fold CV on training years (2008-2023)...")
selected_features, rfecv = run_rfe(X[train_mask], y[train_mask])
print(f"\\nOptimal feature count: {len(selected_features)}")
print("Selected features:")
for f in selected_features:
    print(f"  {f}")
"""),
    new_code_cell("""
# feature importance from RFECV estimator
if hasattr(rfecv, 'estimator_') and hasattr(rfecv.estimator_, 'coef_'):
    coefs = rfecv.estimator_.coef_[0]
    feat_imp = pd.Series(abs(coefs), index=selected_features).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feat_imp.head(20).plot(kind='barh')
    plt.xlabel('|Coefficient| (Logistic Regression)')
    plt.title('Top 20 Features by Coefficient Magnitude (RFECV)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
"""),
]

save_nb(make_notebook(fe_cells), "02_feature_engineering.ipynb")


# notebook 03: model training
mt_cells = [
    new_markdown_cell("# 03 - Model Training\n\nTrain Logistic Regression, Ridge, Random Forest, and LightGBM. Build an ensemble. Evaluate on held-out years 2019, 2022, 2023."),
    new_code_cell("""
import sys, os
sys.path.insert(0, os.path.join('..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_loader import build_matchup_dataset, build_team_stats
from feature_engineering import prepare_training_data, impute_features, run_rfe
from model import train_and_evaluate, build_ensemble, plot_feature_importance, save_eval_summary, TRAIN_YEAR_MAX
"""),
    new_code_cell("""
print("Loading data...")
team_stats = build_team_stats()
matchup_df = build_matchup_dataset(team_stats)
X, y = prepare_training_data(matchup_df)
X = impute_features(X)
years = matchup_df['YEAR'].reset_index(drop=True)

print(f"Total games: {len(X)}")
print(f"Training games (<=2023): {(years <= TRAIN_YEAR_MAX).sum()}")
print(f"Eval games (2019+2022+2023): {years.isin([2019, 2022, 2023]).sum()}")
"""),
    new_code_cell("""
# run RFE on training years
train_mask = years <= TRAIN_YEAR_MAX
print("Running RFE...")
selected_features, rfecv = run_rfe(X[train_mask], y[train_mask])
print(f"Selected {len(selected_features)} features")
"""),
    new_code_cell("""
# train all models
print("Training models...")
results = train_and_evaluate(X, y, years, selected_features=selected_features)
"""),
    new_code_cell("""
# save evaluation summary
summary = save_eval_summary(results)

# print results table
rows = []
for name, res in results.items():
    row = {'model': name, 'cv_logloss': res['cv_logloss']}
    for yr in [2019, 2022, 2023]:
        if yr in res['eval_results']:
            row[f'{yr}_logloss'] = res['eval_results'][yr]['log_loss']
            row[f'{yr}_acc'] = res['eval_results'][yr]['accuracy']
    rows.append(row)
pd.DataFrame(rows).set_index('model')
"""),
    new_code_cell("""
# plot random forest feature importance
rf_model = results.get('random_forest', {}).get('model')
if rf_model is not None:
    plot_feature_importance(rf_model, selected_features)
    img_path = os.path.join('..', 'predictions', 'feature_importance_2026.png')
    if os.path.exists(img_path):
        from IPython.display import Image
        Image(img_path)
"""),
    new_code_cell("""
# build and evaluate ensemble
train_mask_bool = years <= TRAIN_YEAR_MAX
ens, ens_weight = build_ensemble(results, X[train_mask_bool], y[train_mask_bool],
                                  selected_features=selected_features)
print(f"Ensemble ridge weight: {ens_weight:.3f}")
"""),
    new_code_cell("""
# evaluate ensemble on held-out years
from sklearn.metrics import log_loss, accuracy_score

if ens is not None:
    for yr in [2019, 2022, 2023]:
        yr_mask = years == yr
        if yr_mask.sum() == 0:
            continue
        X_eval = X[yr_mask][selected_features].fillna(X[train_mask_bool][selected_features].median())
        y_eval = y[yr_mask]
        probs = ens.predict_proba(X_eval)[:, 1]
        ll = log_loss(y_eval, probs)
        acc = accuracy_score(y_eval, (probs >= 0.5).astype(int))
        print(f"  Ensemble {yr}: log-loss={ll:.4f}, accuracy={acc:.4f}")
"""),
]

save_nb(make_notebook(mt_cells), "03_model_training.ipynb")


# notebook 04: bracket simulation
bs_cells = [
    new_markdown_cell("# 04 - Bracket Simulation\n\nRun 10,000 Monte Carlo simulations of the 2026 NCAA Tournament to generate championship odds per team."),
    new_code_cell("""
import sys, os
sys.path.insert(0, os.path.join('..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from simulate_bracket import simulate_bracket
"""),
    new_code_cell("""
# run full bracket simulation for 2026
odds_df, det_df, upset_df = simulate_bracket(year=2026, n_simulations=10000)
"""),
    new_code_cell("""
# display championship odds
print("Championship Probabilities (Top 20):")
print(odds_df.head(20).to_string(index=False))
"""),
    new_code_cell("""
# bar chart of top-16 championship odds
top16 = odds_df.head(16).copy()
plt.figure(figsize=(12, 5))
bars = plt.barh(range(len(top16)), top16['Championship_Probability'].values)
plt.yticks(range(len(top16)),
           [f"({row['Seed']}) {row['Team']}" for _, row in top16.iterrows()],
           fontsize=9)
plt.xlabel('Championship Probability')
plt.title('2026 NCAA Tournament Championship Odds (10,000 Simulations)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
"""),
    new_code_cell("""
# show predicted bracket (deterministic)
print("Predicted Bracket (Most Likely Winners):")
print(det_df.to_string(index=False))
"""),
    new_code_cell("""
# show upset-adjusted bracket
upsets = upset_df[upset_df.get('Upset_Adjusted', False) == True] if 'Upset_Adjusted' in upset_df.columns else pd.DataFrame()
print(f"Upset-adjusted picks: {len(upsets)} games resolved toward lower seed")
if not upsets.empty:
    print(upsets[['Round', 'Team_A', 'Team_B', 'Team_A_win_prob', 'Predicted_Winner']].to_string(index=False))
"""),
    new_code_cell("""
# load and display saved championship odds CSV
odds_path = os.path.join('..', 'predictions', 'championship_odds_2026.csv')
if os.path.exists(odds_path):
    saved = pd.read_csv(odds_path)
    print(f"Saved championship odds: {odds_path}")
    print(saved.head(10).to_string(index=False))
"""),
]

save_nb(make_notebook(bs_cells), "04_bracket_simulation.ipynb")

print("All notebooks created.")
