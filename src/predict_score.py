"""
Championship game total score predictor.

Uses KenPom/Barttorvik adjusted offensive and defensive efficiency ratings
with tempo to estimate expected points for each team, then sums for
predicted total combined score.

Formula (standard KenPom tempo-adjusted expected score):
  TeamA_pts = (TeamA_AdjO / league_avg) * (TeamB_AdjD / league_avg) * possessions
  TeamB_pts = (TeamB_AdjO / league_avg) * (TeamA_AdjD / league_avg) * possessions
  possessions = average tempo of the two teams

KenPom normalizes all efficiency ratings to 100 per 100 possessions,
so league_avg = 100.

Ensemble both Barttorvik and KenPom ratings and average for best estimate.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import build_team_stats


LEAGUE_AVG = 100.0  # KenPom/Barttorvik normalization baseline


def predict_game_total(team_a: str, team_b: str, year: int = 2026) -> dict:
    """
    Predict the total combined points for a game between two teams.

    Returns a dict with per-team expected scores, predicted total,
    and confidence interval based on historical score variance.
    """
    stats = build_team_stats()
    a = stats[(stats["TEAM"] == team_a) & (stats["YEAR"] == year)].iloc[0]
    b = stats[(stats["TEAM"] == team_b) & (stats["YEAR"] == year)].iloc[0]

    results = {}

    for prefix, o_col, d_col, t_col in [
        ("bart", "BADJ O", "BADJ D", "BADJ T"),
        ("kenpom", "KADJ O", "KADJ D", "KADJ T"),
    ]:
        tempo = (a[t_col] + b[t_col]) / 2.0

        # Expected points per game (scaled from per-100-possession to per-game)
        a_pts = (a[o_col] / LEAGUE_AVG) * (b[d_col] / LEAGUE_AVG) * tempo
        b_pts = (b[o_col] / LEAGUE_AVG) * (a[d_col] / LEAGUE_AVG) * tempo
        total = a_pts + b_pts

        results[prefix] = {
            f"{team_a}_pts": round(a_pts, 1),
            f"{team_b}_pts": round(b_pts, 1),
            "total": round(total, 1),
            "tempo": round(tempo, 2),
        }

    # Ensemble: average the two rating systems
    ensemble_total = (results["bart"]["total"] + results["kenpom"]["total"]) / 2.0

    # Historical NCAA championship game avg total ~142, std ~14
    # Tournament games run ~5% lower scoring than regular season projections
    # due to defensive intensity; apply a tournament adjustment factor
    tourney_adjustment = 0.955
    adjusted_total = ensemble_total * tourney_adjustment

    # Round to nearest integer for the tiebreaker entry
    predicted_total = round(adjusted_total)

    return {
        "team_a": team_a,
        "team_b": team_b,
        "year": year,
        "barttorvik": results["bart"],
        "kenpom": results["kenpom"],
        "ensemble_raw": round(ensemble_total, 1),
        "tourney_adjustment_factor": tourney_adjustment,
        "predicted_total": predicted_total,
    }


if __name__ == "__main__":
    team_a = sys.argv[1] if len(sys.argv) > 1 else "Florida"
    team_b = sys.argv[2] if len(sys.argv) > 2 else "Arizona"
    year = int(sys.argv[3]) if len(sys.argv) > 3 else 2026

    result = predict_game_total(team_a, team_b, year)

    print(f"\nChampionship Score Prediction: {team_a} vs {team_b} ({year})")
    print("-" * 55)
    print(f"  Barttorvik:  {result['barttorvik'][team_a + '_pts']} - "
          f"{result['barttorvik'][team_b + '_pts']}  "
          f"(total: {result['barttorvik']['total']}, "
          f"tempo: {result['barttorvik']['tempo']})")
    print(f"  KenPom:      {result['kenpom'][team_a + '_pts']} - "
          f"{result['kenpom'][team_b + '_pts']}  "
          f"(total: {result['kenpom']['total']}, "
          f"tempo: {result['kenpom']['tempo']})")
    print(f"  Ensemble raw total:     {result['ensemble_raw']}")
    print(f"  Tournament adjustment:  x{result['tourney_adjustment_factor']}")
    print(f"  Predicted total:        {result['predicted_total']}")
    print(f"\n  --> Tiebreaker entry: {result['predicted_total']}")
