import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import build_team_stats, load_tournament_simulation
from feature_engineering import build_features, impute_features

PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "predictions")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# round code -> round name
ROUND_NAMES = {
    64: "Round of 64",
    32: "Round of 32",
    16: "Sweet 16",
    8:  "Elite Eight",
    4:  "Final Four",
    2:  "Championship",
    1:  "Champion",
}

# round advancement: if a team wins round X, they move to round X/2
NEXT_ROUND = {64: 32, 32: 16, 16: 8, 8: 4, 4: 2, 2: 1}


def load_models(model_names=("ridge", "logistic", "lgbm", "random_forest")):
    """load fitted models from disk; return first available."""
    for name in model_names:
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"Loaded model: {name}")
            return model, name
    raise FileNotFoundError(f"No fitted model found in {MODELS_DIR}. Run model.py first.")


def _build_matchup_row(team_a_stats, team_b_stats):
    """
    given two Series of team stats, build a single-row DataFrame
    that matches the format expected by build_features().
    """
    row = {}
    row["SEED_A"] = team_a_stats.get("SEED", 8)
    row["SEED_B"] = team_b_stats.get("SEED", 8)

    for col, val in team_a_stats.items():
        row[f"A_{col}"] = val
    for col, val in team_b_stats.items():
        row[f"B_{col}"] = val

    return pd.DataFrame([row])


def _get_win_prob(model, team_a_stats, team_b_stats):
    """compute P(team A wins) given two team stat dicts."""
    row_df = _build_matchup_row(team_a_stats, team_b_stats)
    features = build_features(row_df)
    features = impute_features(features, strategy="median")

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features)[0, 1]
    else:
        prob = float(features.values[0, 0] > 0)

    # clip to avoid degenerate probabilities
    prob = float(np.clip(prob, 0.01, 0.99))
    return prob


def _parse_bracket(bracket_df, year):
    """
    parse the Tournament Simulation DataFrame for a given year into
    a dict: {by_year_no: {team_no, team, seed, region_no}}.
    also returns the list of first-round matchups as (by_year_no_a, by_year_no_b).
    """
    yr_df = bracket_df[bracket_df["YEAR"] == year].copy()
    yr_df = yr_df.sort_values("BY YEAR NO", ascending=False).reset_index(drop=True)

    teams = {}
    for _, row in yr_df.iterrows():
        teams[int(row["TEAM NO"])] = {
            "team_no": int(row["TEAM NO"]),
            "team": row["TEAM"],
            "seed": int(row["SEED"]),
            "by_year_no": int(row["BY YEAR NO"]),
            "by_round_no": int(row["BY ROUND NO"]) if "BY ROUND NO" in row else 0,
            "current_round": int(row["CURRENT ROUND"]),
        }

    # build first-round matchup pairs: consecutive BY YEAR NO rows
    sorted_teams = yr_df.sort_values("BY YEAR NO", ascending=False).reset_index(drop=True)
    matchup_pairs = []
    for i in range(0, len(sorted_teams) - 1, 2):
        row_a = sorted_teams.iloc[i]
        row_b = sorted_teams.iloc[i + 1]
        if int(row_a["CURRENT ROUND"]) == int(row_b["CURRENT ROUND"]):
            matchup_pairs.append((int(row_a["TEAM NO"]), int(row_b["TEAM NO"])))

    return teams, matchup_pairs


def _simulate_full_bracket(matchup_pairs, team_stats_lookup, model, rng):
    """
    simulate a full tournament bracket once using model win probabilities.
    returns a dict {round_code: [winner_team_no, ...]} and the champion.
    """
    # current bracket: list of team_nos remaining in each position
    # positions are pairs; winner advances to face winner of adjacent pair
    positions = [pair for pair in matchup_pairs]

    round_results = {}
    current_round_code = 64

    while len(positions) > 1 or (len(positions) == 1 and isinstance(positions[0], tuple)):
        next_positions = []
        winners_this_round = []

        for matchup in positions:
            if isinstance(matchup, tuple):
                team_a_no, team_b_no = matchup
            else:
                # already a single team number (bye or placeholder)
                winners_this_round.append(matchup)
                next_positions.append(matchup)
                continue

            # look up stats
            stats_a = team_stats_lookup.get(team_a_no, {})
            stats_b = team_stats_lookup.get(team_b_no, {})

            if not stats_a or not stats_b:
                # fallback: use seed as tiebreaker
                seed_a = stats_a.get("SEED", 8) if stats_a else 8
                seed_b = stats_b.get("SEED", 8) if stats_b else 8
                winner = team_a_no if seed_a <= seed_b else team_b_no
            else:
                p_a = _get_win_prob(model, stats_a, stats_b)
                winner = team_a_no if rng.random() < p_a else team_b_no

            winners_this_round.append(winner)

        round_results[current_round_code] = winners_this_round

        # build next round matchup pairs
        if len(winners_this_round) > 1:
            new_pairs = []
            for i in range(0, len(winners_this_round) - 1, 2):
                new_pairs.append((winners_this_round[i], winners_this_round[i + 1]))
            positions = new_pairs
        else:
            positions = winners_this_round

        current_round_code = NEXT_ROUND.get(current_round_code, 1)
        if current_round_code == 1:
            break

    # handle championship game
    if positions and isinstance(positions[0], tuple):
        team_a_no, team_b_no = positions[0]
        stats_a = team_stats_lookup.get(team_a_no, {})
        stats_b = team_stats_lookup.get(team_b_no, {})
        if stats_a and stats_b:
            p_a = _get_win_prob(model, stats_a, stats_b)
            champion = team_a_no if rng.random() < p_a else team_b_no
        else:
            seed_a = stats_a.get("SEED", 8) if stats_a else 8
            seed_b = stats_b.get("SEED", 8) if stats_b else 8
            champion = team_a_no if seed_a <= seed_b else team_b_no
        round_results[2] = [team_a_no, team_b_no]
        round_results[1] = [champion]
    elif positions:
        if isinstance(positions[0], int):
            round_results[1] = [positions[0]]

    return round_results


def _compute_deterministic_bracket(matchup_pairs, team_stats_lookup, model):
    """
    compute the most-likely bracket by always taking the higher-probability winner.
    returns list of game prediction dicts.
    """
    positions = [pair for pair in matchup_pairs]
    game_results = []
    current_round_code = 64

    while positions:
        next_positions = []
        winners_this_round = []

        for matchup in positions:
            if not isinstance(matchup, tuple):
                winners_this_round.append(matchup)
                next_positions.append(matchup)
                continue

            team_a_no, team_b_no = matchup
            stats_a = team_stats_lookup.get(team_a_no, {})
            stats_b = team_stats_lookup.get(team_b_no, {})

            if stats_a and stats_b:
                p_a = _get_win_prob(model, stats_a, stats_b)
            else:
                seed_a = stats_a.get("SEED", 8) if stats_a else 8
                seed_b = stats_b.get("SEED", 8) if stats_b else 8
                p_a = 0.8 if seed_a < seed_b else 0.2

            winner = team_a_no if p_a >= 0.5 else team_b_no
            loser = team_b_no if winner == team_a_no else team_a_no

            team_a_name = stats_a.get("TEAM", f"Team {team_a_no}")
            team_b_name = stats_b.get("TEAM", f"Team {team_b_no}")
            winner_name = stats_a.get("TEAM") if winner == team_a_no else stats_b.get("TEAM")
            seed_a = stats_a.get("SEED", "?")
            seed_b = stats_b.get("SEED", "?")

            game_results.append({
                "Round": ROUND_NAMES.get(current_round_code, str(current_round_code)),
                "Team_A": f"({seed_a}) {team_a_name}",
                "Team_B": f"({seed_b}) {team_b_name}",
                "Team_A_win_prob": round(p_a, 4),
                "Predicted_Winner": winner_name,
            })

            winners_this_round.append(winner)

        if len(winners_this_round) > 1:
            positions = [(winners_this_round[i], winners_this_round[i + 1])
                         for i in range(0, len(winners_this_round) - 1, 2)]
        else:
            positions = winners_this_round

        current_round_code = NEXT_ROUND.get(current_round_code, 1)
        if current_round_code not in NEXT_ROUND and current_round_code != 1:
            break

    return game_results


def _compute_upset_bracket(matchup_pairs, team_stats_lookup, model, upset_threshold=0.15):
    """
    compute upset-adjusted bracket: if win probability is within threshold of 0.5,
    resolve toward the lower seed (higher seed number).
    """
    positions = [pair for pair in matchup_pairs]
    game_results = []
    current_round_code = 64

    while positions:
        next_positions = []
        winners_this_round = []

        for matchup in positions:
            if not isinstance(matchup, tuple):
                winners_this_round.append(matchup)
                continue

            team_a_no, team_b_no = matchup
            stats_a = team_stats_lookup.get(team_a_no, {})
            stats_b = team_stats_lookup.get(team_b_no, {})

            if stats_a and stats_b:
                p_a = _get_win_prob(model, stats_a, stats_b)
            else:
                seed_a = stats_a.get("SEED", 8) if stats_a else 8
                seed_b = stats_b.get("SEED", 8) if stats_b else 8
                p_a = 0.8 if seed_a < seed_b else 0.2

            seed_a_val = stats_a.get("SEED", 8) if stats_a else 8
            seed_b_val = stats_b.get("SEED", 8) if stats_b else 8

            # if within upset_threshold of 50/50, favor the underdog (higher seed number)
            if abs(p_a - 0.5) <= upset_threshold:
                # resolve toward lower-seeded team (bigger seed number = bigger upset)
                winner = team_a_no if seed_a_val > seed_b_val else team_b_no
            else:
                winner = team_a_no if p_a >= 0.5 else team_b_no

            team_a_name = stats_a.get("TEAM", f"Team {team_a_no}")
            team_b_name = stats_b.get("TEAM", f"Team {team_b_no}")
            winner_name = stats_a.get("TEAM") if winner == team_a_no else stats_b.get("TEAM")
            seed_a = stats_a.get("SEED", "?")
            seed_b = stats_b.get("SEED", "?")

            game_results.append({
                "Round": ROUND_NAMES.get(current_round_code, str(current_round_code)),
                "Team_A": f"({seed_a}) {team_a_name}",
                "Team_B": f"({seed_b}) {team_b_name}",
                "Team_A_win_prob": round(p_a, 4),
                "Predicted_Winner": winner_name,
                "Upset_Adjusted": abs(p_a - 0.5) <= upset_threshold,
            })

            winners_this_round.append(winner)

        if len(winners_this_round) > 1:
            positions = [(winners_this_round[i], winners_this_round[i + 1])
                         for i in range(0, len(winners_this_round) - 1, 2)]
        else:
            positions = winners_this_round

        current_round_code = NEXT_ROUND.get(current_round_code, 1)
        if current_round_code not in NEXT_ROUND and current_round_code != 1:
            break

    return game_results


def run_monte_carlo(matchup_pairs, team_stats_lookup, model, n_simulations=10000, seed=42):
    """
    run n_simulations full tournament bracket simulations.
    returns championship_counts: {team_no: count_of_championships}
    """
    rng = np.random.default_rng(seed)
    championship_counts = {}
    print(f"Running {n_simulations} Monte Carlo simulations...")

    for sim_i in range(n_simulations):
        if sim_i % 1000 == 0 and sim_i > 0:
            print(f"  Simulation {sim_i}/{n_simulations}...")
        round_results = _simulate_full_bracket(matchup_pairs, team_stats_lookup, model, rng)
        champion_list = round_results.get(1, [])
        if champion_list:
            champ = champion_list[0]
            championship_counts[champ] = championship_counts.get(champ, 0) + 1

    return championship_counts


def simulate_bracket(year=None, n_simulations=10000):
    """
    main entry point for bracket simulation.
    loads the bracket, runs Monte Carlo, and saves outputs.
    """
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # load current bracket
    bracket_df = load_tournament_simulation()
    if year is None:
        year = int(bracket_df["YEAR"].max())
    print(f"Simulating bracket for year: {year}")

    teams, matchup_pairs = _parse_bracket(bracket_df, year)
    team_nos = list(teams.keys())

    # load team stats for the current year
    team_stats = build_team_stats()
    year_stats = team_stats[team_stats["YEAR"] == year]

    # build lookup: team_no -> stats dict
    team_stats_lookup = {}
    for _, row in year_stats.iterrows():
        team_no = int(row["TEAM NO"])
        if team_no in teams:
            d = row.to_dict()
            d["SEED"] = teams[team_no]["seed"]
            d["TEAM"] = teams[team_no]["team"]
            team_stats_lookup[team_no] = d

    # fill in any missing teams with seed-based fallback
    for team_no, info in teams.items():
        if team_no not in team_stats_lookup:
            team_stats_lookup[team_no] = {
                "TEAM": info["team"],
                "SEED": info["seed"],
                "TEAM NO": team_no,
            }

    # load model
    model, model_name = load_models()

    # run Monte Carlo simulations
    champ_counts = run_monte_carlo(matchup_pairs, team_stats_lookup, model, n_simulations=n_simulations)

    # build championship odds DataFrame
    odds_rows = []
    for team_no, info in teams.items():
        count = champ_counts.get(team_no, 0)
        odds_rows.append({
            "Team": info["team"],
            "Seed": info["seed"],
            "Championship_Wins": count,
            "Championship_Probability": round(count / n_simulations, 4),
        })
    odds_df = pd.DataFrame(odds_rows).sort_values("Championship_Probability", ascending=False).reset_index(drop=True)
    odds_path = os.path.join(PREDICTIONS_DIR, f"championship_odds_{year}.csv")
    odds_df.to_csv(odds_path, index=False)
    print(f"Championship odds saved to {odds_path}")

    # compute deterministic bracket
    det_games = _compute_deterministic_bracket(matchup_pairs, team_stats_lookup, model)
    det_df = pd.DataFrame(det_games)
    det_path = os.path.join(PREDICTIONS_DIR, f"bracket_{year}.csv")
    det_df.to_csv(det_path, index=False)
    print(f"Deterministic bracket saved to {det_path}")

    # compute upset-adjusted bracket
    upset_games = _compute_upset_bracket(matchup_pairs, team_stats_lookup, model)
    upset_df = pd.DataFrame(upset_games)
    upset_path = os.path.join(PREDICTIONS_DIR, f"bracket_upset_adjusted_{year}.csv")
    upset_df.to_csv(upset_path, index=False)
    print(f"Upset-adjusted bracket saved to {upset_path}")

    # print top-10 championship odds
    print(f"\nTop 10 championship contenders ({year}):")
    print(odds_df.head(10).to_string(index=False))

    return odds_df, det_df, upset_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NCAA Tournament bracket simulation")
    parser.add_argument("--year", type=int, default=None, help="Tournament year (default: latest)")
    parser.add_argument("--sims", type=int, default=10000, help="Number of Monte Carlo simulations")
    args = parser.parse_args()

    odds_df, det_df, upset_df = simulate_bracket(year=args.year, n_simulations=args.sims)
