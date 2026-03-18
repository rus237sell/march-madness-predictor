"""
Kristin's Challenge — predict the number of overtime games in
Thursday + Friday Round of 64 games.

Methodology:
  - Build a logistic regression OT-probability model from 17 years of
    tournament historical game pairs.
  - OT signal: games where the final margin is exactly 1 point (the
    clearest indicator of an overtime or near-overtime finish).
  - Features per game:
      abs_em_diff      : |AdjEM_A - AdjEM_B| (tighter = more likely OT)
      avg_tempo        : average possessions per game (slower = less scoring variance)
      seed_diff_abs    : |seed_A - seed_B| (tight matchups)
      ft_pct_avg       : average FT% of both teams (clutch execution)
      ps_em_change_avg : average KenPom trajectory (peaking teams play tighter)
  - Apply to all Round of 64 2026 matchups (Thursday + Friday).
  - Sum P(OT) across all games to get E[# OT games].

Usage:
    python3 src/predict_ot.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import build_team_stats, load_tournament_matchups


# ── helpers ────────────────────────────────────────────────────────────────────────────
def _pair_games(matchups: pd.DataFrame, require_scores: bool = True) -> pd.DataFrame:
    """
    Reconstruct game-level records from the team-level tournament matchup rows.
    Returns one row per game with scores, seeds, and identifiers.

    Set require_scores=False to include future/unplayed matchups (no SCORE).
    """
    rows = []
    for yr in sorted(matchups["YEAR"].unique()):
        yr_df = matchups[matchups["YEAR"] == yr].sort_values("BY YEAR NO").reset_index(drop=True)
        for rnd in yr_df["CURRENT ROUND"].unique():
            rnd_df = yr_df[yr_df["CURRENT ROUND"] == rnd].sort_values("BY YEAR NO").reset_index(drop=True)
            i = 0
            while i < len(rnd_df) - 1:
                a, b = rnd_df.iloc[i], rnd_df.iloc[i + 1]
                has_scores = pd.notna(a["SCORE"]) and pd.notna(b["SCORE"])
                close_idx  = abs(int(a["BY YEAR NO"]) - int(b["BY YEAR NO"])) <= 2
                if (not require_scores or has_scores) and close_idx:
                    combined = (a["SCORE"] + b["SCORE"]) if has_scores else None
                    margin   = abs(a["SCORE"] - b["SCORE"]) if has_scores else None
                    rows.append({
                        "year":      int(yr),
                        "round":     int(rnd),
                        "team_a":    a["TEAM"],
                        "seed_a":    int(a["SEED"]),
                        "team_no_a": int(a["TEAM NO"]),
                        "team_b":    b["TEAM"],
                        "seed_b":    int(b["SEED"]),
                        "team_no_b": int(b["TEAM NO"]),
                        "score_a":   a["SCORE"] if has_scores else None,
                        "score_b":   b["SCORE"] if has_scores else None,
                        "combined":  combined,
                        "margin":    margin,
                        "ot":        int(margin == 1) if margin is not None else None,
                    })
                    i += 2
                else:
                    i += 1
    return pd.DataFrame(rows)
def _build_ot_features(game_df: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Merge per-team efficiency stats onto game pairs and compute
    game-level features for OT prediction.
    """
    stat_cols = ["YEAR", "TEAM NO", "TEAM", "BADJ EM", "KADJ O", "KADJ D",
                 "KADJ T", "FT%", "PS_EM_CHANGE", "AP_RANK_FINAL",
                 "3PT%", "3PTR", "EFG%", "EFG%D"]
    avail = [c for c in stat_cols if c in team_stats.columns]
    # deduplicate: team stats appear twice (once per each tournament game row)
    ts = team_stats[avail].drop_duplicates(subset=["YEAR", "TEAM NO"]).copy()

    # build a clean lookup: lowercase key columns to match game_df conventions
    stat_only = [c for c in avail if c not in ("YEAR", "TEAM NO", "TEAM")]
    ts_lookup = ts[["YEAR", "TEAM NO"] + stat_only].copy()
    ts_lookup = ts_lookup.rename(columns={"YEAR": "year", "TEAM NO": "team_no"})

    def _prefix_rename(prefix):
        return {c: f"{prefix}_{c.lower().replace(' ', '_').replace('%', 'pct').replace('-', '_')}"
                for c in stat_only}

    g = game_df.copy()
    # merge team A stats
    ts_a = ts_lookup.rename(columns={"team_no": "_team_no_a", **_prefix_rename("a")})
    g = g.merge(ts_a, left_on=["year", "team_no_a"], right_on=["year", "_team_no_a"], how="left")
    g = g.drop(columns=["_team_no_a"], errors="ignore")

    # merge team B stats
    ts_b = ts_lookup.rename(columns={"team_no": "_team_no_b", **_prefix_rename("b")})
    g = g.merge(ts_b, left_on=["year", "team_no_b"], right_on=["year", "_team_no_b"], how="left")
    g = g.drop(columns=["_team_no_b"], errors="ignore")

    feats = pd.DataFrame(index=g.index)
    feats["abs_em_diff"]      = (g["a_badj_em"] - g["b_badj_em"]).abs()
    feats["avg_tempo"]        = (g["a_kadj_t"] + g["b_kadj_t"]) / 2.0
    feats["seed_diff_abs"]    = (g["seed_a"] - g["seed_b"]).abs()
    # FT% column: FT% → a_ftpct after the rename transform
    feats["ft_pct_avg"]       = (g.get("a_ftpct", pd.Series(0.72, index=g.index)) +
                                  g.get("b_ftpct", pd.Series(0.72, index=g.index))) / 2.0
    feats["ps_change_avg"]    = (g["a_ps_em_change"].fillna(0) + g["b_ps_em_change"].fillna(0)) / 2.0
    feats["both_ranked"]      = (
        (g["a_ap_rank_final"].fillna(26) < 26) & (g["b_ap_rank_final"].fillna(26) < 26)
    ).astype(float)
    a_vol = g.get("a_3ptr", pd.Series(30, index=g.index)).fillna(30) * \
            (1.0 - g.get("a_3ptpct", pd.Series(0.33, index=g.index)).fillna(0.33))
    b_vol = g.get("b_3ptr", pd.Series(30, index=g.index)).fillna(30) * \
            (1.0 - g.get("b_3ptpct", pd.Series(0.33, index=g.index)).fillna(0.33))
    feats["avg_3pt_volatility"] = (a_vol + b_vol) / 2.0

    return feats, g
def train_ot_model(games: pd.DataFrame, team_stats: pd.DataFrame):
    """
    Fit an OT-probability model on historical tournament games.

    Because OT is rare (~4.8%) and signal is weak, we use a two-stage approach:
      1. Logistic regression with weak regularization to rank games by OT likelihood.
      2. Platt-style calibration (isotonic regression via CalibratedClassifierCV)
         to ensure predicted probabilities are anchored to the historical base rate.

    This prevents the model from predicting absurdly high OT rates.
    Returns (model, scaler, feature_names, training_stats).
    """
    feats, _ = _build_ot_features(games, team_stats)
    y = games["ot"].reset_index(drop=True).values

    feats = feats.reset_index(drop=True)
    valid = feats.notna().all(axis=1).values
    X = feats[valid].values
    y = y[valid]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Train logistic with NO class weighting so probabilities stay near base rate
    base_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    # Wrap with isotonic calibration to improve probability estimates
    model = CalibratedClassifierCV(base_model, cv=5, method="isotonic")
    model.fit(X_s, y)

    # Verify calibration: mean predicted prob should ≈ historical OT rate
    mean_pred = model.predict_proba(X_s)[:, 1].mean()

    cv_scores = cross_val_score(
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        X_s, y, cv=5, scoring="roc_auc"
    )

    return model, scaler, feats.columns.tolist(), {
        "n_games":         len(y),
        "ot_rate":         float(y.mean()),
        "ot_count":        int(y.sum()),
        "mean_pred_prob":  round(float(mean_pred), 4),
        "cv_auc_mean":     float(cv_scores.mean()),
        "cv_auc_std":      float(cv_scores.std()),
    }
def predict_tournament_ot(year: int = 2026) -> dict:
    """
    Predict expected number of OT games for all Round of 64 games
    (Thursday + Friday) in the given tournament year.

    Returns a dict with per-game OT probabilities and summary stats.
    """
    matchups   = load_tournament_matchups()
    team_stats = build_team_stats()

    # Build historical OT model from all years with completed scores
    historical = _pair_games(matchups[matchups["SCORE"].notna()])
    r64_hist   = historical[historical["round"] == 64].copy()

    model, scaler, feat_names, train_stats = train_ot_model(r64_hist, team_stats)

    # Get 2026 Round of 64 matchups (SCORE is NaN = not yet played)
    current = _pair_games(matchups[(matchups["YEAR"] == year) &
                                   (matchups["CURRENT ROUND"] == 64)],
                          require_scores=False)

    feats_2026, games_2026 = _build_ot_features(current, team_stats)

    # Impute any missing values with training-set means
    for col in feat_names:
        if col in feats_2026.columns:
            feats_2026[col] = feats_2026[col].fillna(feats_2026[col].median())

    X_2026 = scaler.transform(feats_2026[feat_names].values)
    probs   = model.predict_proba(X_2026)[:, 1]

    results = []
    for i, (_, row) in enumerate(games_2026.iterrows()):
        results.append({
            "game":          f"({row['seed_a']}) {row['team_a']} vs ({row['seed_b']}) {row['team_b']}",
            "team_a":        row["team_a"],
            "seed_a":        int(row["seed_a"]),
            "team_b":        row["team_b"],
            "seed_b":        int(row["seed_b"]),
            "ot_probability": round(float(probs[i]), 4),
        })

    results_df = pd.DataFrame(results).sort_values("ot_probability", ascending=False)
    expected_ot = float(probs.sum())
    predicted_ot = round(expected_ot)

    return {
        "year":            year,
        "round":           "Round of 64 (Thursday + Friday)",
        "model_training":  train_stats,
        "n_games":         len(results),
        "expected_ot":     round(expected_ot, 2),
        "predicted_ot":    predicted_ot,
        "games":           results_df.to_dict(orient="records"),
    }
if __name__ == "__main__":
    import json
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026

    print(f"\nKristin's Challenge — OT Prediction for {year} Round of 64")
    print("=" * 62)

    result = predict_tournament_ot(year)
    ts = result["model_training"]

    print(f"\nModel trained on: {ts['n_games']} historical R64 games")
    print(f"  Historical OT rate: {ts['ot_rate']:.2%}  ({ts['ot_count']} OT games)")
    print(f"  Cross-val AUC: {ts['cv_auc_mean']:.3f} ± {ts['cv_auc_std']:.3f}")
    print(f"\n# of games: {result['n_games']}  (Thursday + Friday, Round of 64)")
    print(f"Expected OT games: {result['expected_ot']}")
    print(f"\n>>> PREDICTION: {result['predicted_ot']} OT game(s) on Thursday + Friday <<<")

    print("\nTop 10 most likely OT games:")
    print("-" * 62)
    for r in result["games"][:10]:
        print(f"  {r['ot_probability']:.2%}  {r['game']}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "predictions",
                            f"kristins_challenge_{year}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull results saved to predictions/kristins_challenge_{year}.json")
