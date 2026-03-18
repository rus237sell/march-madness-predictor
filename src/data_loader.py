import os
import pandas as pd
import numpy as np

# default path relative to repo root
DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DATA_PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def _raw(filename):
    return os.path.join(DATA_RAW, filename)


def load_kenpom_barttorvik():
    df = pd.read_csv(_raw("KenPom Barttorvik.csv"))
    df.columns = df.columns.str.strip()
    return df


def load_tournament_matchups():
    df = pd.read_csv(_raw("Tournament Matchups.csv"))
    df.columns = df.columns.str.strip()
    return df


def load_resumes():
    df = pd.read_csv(_raw("Resumes.csv"))
    df.columns = df.columns.str.strip()
    return df


def load_shooting_splits():
    df = pd.read_csv(_raw("Shooting Splits.csv"))
    df.columns = df.columns.str.strip()
    return df


def load_teamsheet_ranks():
    df = pd.read_csv(_raw("Teamsheet Ranks.csv"))
    df.columns = df.columns.str.strip()
    return df


def load_barttorvik_neutral():
    df = pd.read_csv(_raw("Barttorvik Away-Neutral.csv"))
    df.columns = df.columns.str.strip()
    return df


def load_evan_miya():
    df = pd.read_csv(_raw("EvanMiya.csv"))
    df.columns = df.columns.str.strip()
    return df


def load_538():
    df = pd.read_csv(_raw("538 Ratings.csv"))
    df.columns = df.columns.str.strip()
    return df


def load_conference_stats():
    df = pd.read_csv(_raw("Conference Stats.csv"))
    df.columns = df.columns.str.strip()
    return df


def load_conference_results():
    df = pd.read_csv(_raw("Conference Results.csv"))
    df.columns = df.columns.str.strip()
    return df


def load_team_results():
    df = pd.read_csv(_raw("Team Results.csv"))
    df.columns = df.columns.str.strip()
    return df


def load_rppf_ratings():
    df = pd.read_csv(_raw("RPPF Ratings.csv"))
    df.columns = df.columns.str.strip()
    return df


def load_z_ratings():
    df = pd.read_csv(_raw("Z Rating Teams.csv"))
    df.columns = df.columns.str.strip()
    return df


def load_heat_check():
    df = pd.read_csv(_raw("Heat Check Tournament Index.csv"))
    df.columns = df.columns.str.strip()
    return df


def load_tournament_simulation():
    df = pd.read_csv(_raw("Tournament Simulation.csv"))
    df.columns = df.columns.str.strip()
    return df


def _build_game_pairs(matchups):
    # each game is represented by two consecutive rows sharing a CURRENT ROUND
    # within the same year; pairs are formed by sorting on BY YEAR NO descending
    # and grouping consecutive rows
    rows = []
    matchups_sorted = matchups.sort_values(["YEAR", "BY YEAR NO"], ascending=[True, False]).reset_index(drop=True)
    i = 0
    while i < len(matchups_sorted) - 1:
        row_a = matchups_sorted.iloc[i]
        row_b = matchups_sorted.iloc[i + 1]
        # confirm they're in the same year and same current round
        if row_a["YEAR"] == row_b["YEAR"] and row_a["CURRENT ROUND"] == row_b["CURRENT ROUND"]:
            # the winner is the team whose ROUND < CURRENT ROUND (they advanced)
            # if both have ROUND == CURRENT ROUND or both advanced, handle edge case
            winner_a = int(row_a["ROUND"]) < int(row_a["CURRENT ROUND"])
            winner_b = int(row_b["ROUND"]) < int(row_b["CURRENT ROUND"])
            if winner_a:
                team_w, team_l = row_a, row_b
            elif winner_b:
                team_w, team_l = row_b, row_a
            else:
                # game not yet played (current year in progress), use row_a as team_a
                team_w, team_l = row_a, row_b
                rows.append({
                    "YEAR": int(row_a["YEAR"]),
                    "CURRENT_ROUND": int(row_a["CURRENT ROUND"]),
                    "TEAM_A_NO": int(team_w["TEAM NO"]),
                    "TEAM_A": team_w["TEAM"],
                    "SEED_A": int(team_w["SEED"]),
                    "TEAM_B_NO": int(team_l["TEAM NO"]),
                    "TEAM_B": team_l["TEAM"],
                    "SEED_B": int(team_l["SEED"]),
                    "TEAM_A_WIN": None,
                })
                i += 2
                continue
            rows.append({
                "YEAR": int(row_a["YEAR"]),
                "CURRENT_ROUND": int(row_a["CURRENT ROUND"]),
                "TEAM_A_NO": int(team_w["TEAM NO"]),
                "TEAM_A": team_w["TEAM"],
                "SEED_A": int(team_w["SEED"]),
                "TEAM_B_NO": int(team_l["TEAM NO"]),
                "TEAM_B": team_l["TEAM"],
                "SEED_B": int(team_l["SEED"]),
                "TEAM_A_WIN": 1,
            })
            i += 2
        else:
            i += 1
    return pd.DataFrame(rows)


def _build_team_stats(kb, resumes, shooting, teamsheet, neutral, evan, z_rat, conf_stats, rppf):
    # start with KenPom Barttorvik as base
    stats = kb.copy()

    # merge resumes (Q1 wins, resume score, ELO, etc.)
    res_cols = ["YEAR", "TEAM NO", "Q1 W", "Q2 W", "Q1 PLUS Q2 W", "Q3 Q4 L", "NET RPI", "RESUME", "ELO", "B POWER"]
    res_avail = [c for c in res_cols if c in resumes.columns]
    stats = stats.merge(resumes[res_avail].rename(columns={"Q1 W": "Q1_W", "Q2 W": "Q2_W",
                                                             "Q1 PLUS Q2 W": "Q1Q2_W",
                                                             "Q3 Q4 L": "Q3Q4_L",
                                                             "NET RPI": "NET_RPI",
                                                             "B POWER": "B_POWER"}),
                        on=["YEAR", "TEAM NO"], how="left")

    # merge shooting splits
    sh_cols = ["YEAR", "TEAM NO", "DUNKS FG%", "DUNKS SHARE", "CLOSE TWOS FG%", "CLOSE TWOS SHARE",
               "FARTHER TWOS FG%", "FARTHER TWOS SHARE", "THREES FG%", "THREES SHARE",
               "THREES FG%D", "THREES D SHARE"]
    sh_avail = [c for c in sh_cols if c in shooting.columns]
    stats = stats.merge(shooting[sh_avail], on=["YEAR", "TEAM NO"], how="left")

    # merge teamsheet quad records where available
    if teamsheet is not None and not teamsheet.empty:
        ts_cols = ["YEAR", "TEAM NO", "Q1A W", "Q1A L", "Q1 W", "Q1 L", "Q1&2 W", "Q1&2 L"]
        ts_avail = [c for c in ts_cols if c in teamsheet.columns]
        ts_rename = {"Q1A W": "Q1A_W", "Q1A L": "Q1A_L",
                     "Q1 W": "TS_Q1_W", "Q1 L": "TS_Q1_L",
                     "Q1&2 W": "TS_Q1Q2_W", "Q1&2 L": "TS_Q1Q2_L"}
        stats = stats.merge(teamsheet[ts_avail].rename(columns=ts_rename),
                            on=["YEAR", "TEAM NO"], how="left")

    # merge Barttorvik neutral/away stats for neutral-site adjusted metrics
    if neutral is not None and not neutral.empty:
        nt_cols = ["YEAR", "TEAM NO", "BADJ EM", "BADJ O", "BADJ D", "BARTHAG", "EFG%", "EFG%D",
                   "TOV%", "TOV%D", "OREB%", "DREB%", "FT%", "PPPO", "PPPD"]
        nt_avail = [c for c in nt_cols if c in neutral.columns]
        nt_rename = {c: f"NT_{c.replace(' ', '_').replace('%', 'PCT')}" for c in nt_avail if c not in ["YEAR", "TEAM NO"]}
        stats = stats.merge(neutral[nt_avail].rename(columns=nt_rename),
                            on=["YEAR", "TEAM NO"], how="left")

    # merge EvanMiya ratings
    if evan is not None and not evan.empty:
        em_cols = ["YEAR", "TEAM NO", "O RATE", "D RATE", "RELATIVE RATING", "TRUE TEMPO", "KILLSHOTS MARGIN"]
        em_avail = [c for c in em_cols if c in evan.columns]
        em_rename = {"O RATE": "EM_O_RATE", "D RATE": "EM_D_RATE",
                     "RELATIVE RATING": "EM_RATING", "TRUE TEMPO": "EM_TEMPO",
                     "KILLSHOTS MARGIN": "EM_KILLSHOTS"}
        stats = stats.merge(evan[em_avail].rename(columns=em_rename),
                            on=["YEAR", "TEAM NO"], how="left")

    # merge Z ratings
    if z_rat is not None and not z_rat.empty:
        z_cols = ["YEAR", "TEAM NO", "Z RATING", "Z RATING RANK"]
        z_avail = [c for c in z_cols if c in z_rat.columns]
        z_rename = {"Z RATING": "Z_RATING", "Z RATING RANK": "Z_RATING_RANK"}
        stats = stats.merge(z_rat[z_avail].rename(columns=z_rename),
                            on=["YEAR", "TEAM NO"], how="left")

    # merge RPPF ratings
    if rppf is not None and not rppf.empty:
        r_cols = [c for c in rppf.columns if c in ["YEAR", "TEAM NO", "RPPF", "RPPF RANK"]]
        if len(r_cols) > 2:
            stats = stats.merge(rppf[r_cols], on=["YEAR", "TEAM NO"], how="left")

    return stats


def _add_historical_program_features(stats, team_results):
    # compute historical program tournament win rate and advancement rates since 2003
    tr = team_results.copy()
    tr.columns = tr.columns.str.strip()

    # compute win rate, S16 rate, F4 rate, and champ rate from career totals
    prog = tr[["TEAM ID", "TEAM", "GAMES", "W", "S16", "F4", "CHAMP"]].copy()
    prog = prog.rename(columns={"TEAM ID": "TEAM_ID_HIST"})
    prog["HIST_WIN_RATE"] = prog["W"] / prog["GAMES"].replace(0, np.nan)
    prog["HIST_S16_RATE"] = prog["S16"] / prog["GAMES"].replace(0, np.nan)
    prog["HIST_F4_RATE"] = prog["F4"] / prog["GAMES"].replace(0, np.nan)
    prog["HIST_CHAMP_RATE"] = prog["CHAMP"] / prog["GAMES"].replace(0, np.nan)

    blue_bloods = {"Duke", "Kansas", "Kentucky", "North Carolina", "UConn",
                   "Connecticut", "Arizona", "UCLA"}
    prog["BLUE_BLOOD"] = prog["TEAM"].isin(blue_bloods).astype(int)

    prog = prog[["TEAM", "HIST_WIN_RATE", "HIST_S16_RATE", "HIST_F4_RATE", "HIST_CHAMP_RATE", "BLUE_BLOOD"]]

    # join on team name - use fuzzy merge via team name string
    stats = stats.merge(prog, on="TEAM", how="left")
    stats["BLUE_BLOOD"] = stats["BLUE_BLOOD"].fillna(0).astype(int)
    return stats


def _add_conference_features(stats, conf_stats):
    # conference average AdjEM as a conference strength signal per year
    kb_conf = stats[["YEAR", "CONF", "BADJ EM"]].copy()
    conf_avg = kb_conf.groupby(["YEAR", "CONF"])["BADJ EM"].mean().reset_index()
    conf_avg.columns = ["YEAR", "CONF", "CONF_AVG_ADJEM"]
    stats = stats.merge(conf_avg, on=["YEAR", "CONF"], how="left")
    return stats


def build_team_stats():
    """load and merge all team-level data sources into a single wide DataFrame."""
    kb = load_kenpom_barttorvik()
    resumes = load_resumes()
    shooting = load_shooting_splits()
    teamsheet = load_teamsheet_ranks()
    neutral = load_barttorvik_neutral()
    evan = load_evan_miya()
    z_rat = load_z_ratings()
    conf_stats = load_conference_stats()
    team_results = load_team_results()
    rppf = load_rppf_ratings()

    stats = _build_team_stats(kb, resumes, shooting, teamsheet, neutral, evan, z_rat, conf_stats, rppf)
    stats = _add_historical_program_features(stats, team_results)
    stats = _add_conference_features(stats, conf_stats)

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    stats.to_csv(os.path.join(DATA_PROCESSED, "team_stats.csv"), index=False)
    return stats


def build_matchup_dataset(team_stats=None):
    """
    build labeled matchup dataset: each row is a game with delta features between
    team A (winner) and team B (loser). target = 1 always (winner is team A) but
    we duplicate with flipped sign to create balanced training data.
    """
    if team_stats is None:
        team_stats = build_team_stats()

    matchups = load_tournament_matchups()
    games = _build_game_pairs(matchups)
    games_train = games[games["TEAM_A_WIN"] == 1].copy()

    # rename stats columns for merging
    # exclude identifier columns from the feature set that gets A_/B_ prefixed
    id_cols = {"YEAR", "CONF", "CONF ID", "QUAD NO", "QUAD ID",
               "TEAM NO", "TEAM ID", "TEAM", "SEED", "ROUND", "CURRENT ROUND"}
    stat_cols = [c for c in team_stats.columns if c not in id_cols]
    team_a_stats = team_stats.rename(columns={c: f"A_{c}" for c in stat_cols})
    team_b_stats = team_stats.rename(columns={c: f"B_{c}" for c in stat_cols})

    # merge team A stats
    merged = games_train.merge(
        team_a_stats[["YEAR", "TEAM NO"] + [f"A_{c}" for c in stat_cols]],
        left_on=["YEAR", "TEAM_A_NO"], right_on=["YEAR", "TEAM NO"], how="left"
    ).drop(columns=["TEAM NO"])

    # merge team B stats
    merged = merged.merge(
        team_b_stats[["YEAR", "TEAM NO"] + [f"B_{c}" for c in stat_cols]],
        left_on=["YEAR", "TEAM_B_NO"], right_on=["YEAR", "TEAM NO"], how="left"
    ).drop(columns=["TEAM NO"])

    # create flipped version so the model sees both perspectives
    flipped = merged.copy()
    a_cols_all = [c for c in merged.columns if c.startswith("A_")]
    b_cols_all = [c for c in merged.columns if c.startswith("B_")]
    for ac, bc in zip(a_cols_all, b_cols_all):
        flipped[ac] = merged[bc]
        flipped[bc] = merged[ac]
    flipped["TEAM_A_WIN"] = 0
    flipped[["TEAM_A_NO", "TEAM_A", "SEED_A", "TEAM_B_NO", "TEAM_B", "SEED_B"]] = merged[
        ["TEAM_B_NO", "TEAM_B", "SEED_B", "TEAM_A_NO", "TEAM_A", "SEED_A"]
    ].values

    full = pd.concat([merged, flipped], ignore_index=True)
    full.to_csv(os.path.join(DATA_PROCESSED, "matchup_dataset.csv"), index=False)
    return full


def load_current_bracket():
    """load the current year tournament bracket for simulation."""
    sim = load_tournament_simulation()
    return sim


if __name__ == "__main__":
    print("Building team stats...")
    ts = build_team_stats()
    print(f"Team stats shape: {ts.shape}")

    print("Building matchup dataset...")
    md = build_matchup_dataset(ts)
    print(f"Matchup dataset shape: {md.shape}")
    print(f"Years: {sorted(md['YEAR'].unique())}")
