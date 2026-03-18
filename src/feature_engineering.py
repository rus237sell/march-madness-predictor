import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# columns that identify a matchup row, not features
META_COLS = ["YEAR", "CURRENT_ROUND", "TEAM_A_NO", "TEAM_A", "SEED_A",
             "TEAM_B_NO", "TEAM_B", "SEED_B", "TEAM_A_WIN"]


def _delta(df, col_a, col_b, name):
    # compute delta between two columns and return as Series
    return (df[col_a] - df[col_b]).rename(name)


def _safe(df, col, default=np.nan):
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def build_features(df):
    """
    given a matchup DataFrame with A_ and B_ prefixed stat columns,
    compute all engineered features and return a feature matrix.
    """
    feats = pd.DataFrame(index=df.index)

    # efficiency and scoring features
    feats["delta_badj_em"] = _safe(df, "A_BADJ EM") - _safe(df, "B_BADJ EM")
    feats["a_kadj_o"] = _safe(df, "A_KADJ O")
    feats["b_kadj_o"] = _safe(df, "B_KADJ O")
    feats["a_kadj_d"] = _safe(df, "A_KADJ D")
    feats["b_kadj_d"] = _safe(df, "B_KADJ D")
    feats["delta_kadj_em"] = _safe(df, "A_KADJ EM") - _safe(df, "B_KADJ EM")
    feats["delta_barthag"] = _safe(df, "A_BARTHAG") - _safe(df, "B_BARTHAG")

    # efficiency ratios: AdjO / AdjD per team then delta
    a_eff_ratio = _safe(df, "A_KADJ O") / _safe(df, "A_KADJ D").replace(0, np.nan)
    b_eff_ratio = _safe(df, "B_KADJ O") / _safe(df, "B_KADJ D").replace(0, np.nan)
    feats["a_eff_ratio"] = a_eff_ratio
    feats["b_eff_ratio"] = b_eff_ratio
    feats["delta_eff_ratio"] = a_eff_ratio - b_eff_ratio

    # points per possession
    feats["a_pppo"] = _safe(df, "A_PPPO")
    feats["b_pppo"] = _safe(df, "B_PPPO")
    feats["a_pppd"] = _safe(df, "A_PPPD")
    feats["b_pppd"] = _safe(df, "B_PPPD")
    feats["delta_pppo"] = _safe(df, "A_PPPO") - _safe(df, "B_PPPO")
    feats["delta_pppd"] = _safe(df, "A_PPPD") - _safe(df, "B_PPPD")

    # four factors
    feats["delta_efg_off"] = _safe(df, "A_EFG%") - _safe(df, "B_EFG%")
    feats["delta_efg_def"] = _safe(df, "A_EFG%D") - _safe(df, "B_EFG%D")
    feats["delta_tov_off"] = _safe(df, "A_TOV%") - _safe(df, "B_TOV%")
    feats["delta_tov_def"] = _safe(df, "A_TOV%D") - _safe(df, "B_TOV%D")
    feats["delta_oreb"] = _safe(df, "A_OREB%") - _safe(df, "B_OREB%")
    feats["delta_dreb"] = _safe(df, "A_DREB%") - _safe(df, "B_DREB%")
    feats["delta_ftr_off"] = _safe(df, "A_FTR") - _safe(df, "B_FTR")
    feats["delta_ftr_def"] = _safe(df, "A_FTRD") - _safe(df, "B_FTRD")
    feats["a_ft_pct"] = _safe(df, "A_FT%")
    feats["b_ft_pct"] = _safe(df, "B_FT%")
    feats["delta_ft_pct"] = _safe(df, "A_FT%") - _safe(df, "B_FT%")

    # shooting profile
    feats["delta_3pt_rate"] = _safe(df, "A_3PTR") - _safe(df, "B_3PTR")
    feats["delta_3pt_rate_def"] = _safe(df, "A_3PTRD") - _safe(df, "B_3PTRD")
    feats["delta_3pt_pct"] = _safe(df, "A_3PT%") - _safe(df, "B_3PT%")
    feats["delta_3pt_pct_def"] = _safe(df, "A_3PT%D") - _safe(df, "B_3PT%D")
    feats["delta_2pt_pct"] = _safe(df, "A_2PT%") - _safe(df, "B_2PT%")
    feats["delta_2pt_pct_def"] = _safe(df, "A_2PT%D") - _safe(df, "B_2PT%D")
    feats["delta_2pt_rate"] = _safe(df, "A_2PTR") - _safe(df, "B_2PTR")
    feats["delta_ts_proxy"] = feats["delta_efg_off"]

    a_mid = 1.0 - _safe(df, "A_3PTR", 0) / 100.0 - _safe(df, "A_CLOSE TWOS SHARE", 0) / 100.0
    b_mid = 1.0 - _safe(df, "B_3PTR", 0) / 100.0 - _safe(df, "B_CLOSE TWOS SHARE", 0) / 100.0
    feats["a_mid_rate"] = a_mid
    feats["b_mid_rate"] = b_mid
    feats["delta_mid_rate"] = a_mid - b_mid

    # tempo
    feats["delta_tempo"] = _safe(df, "A_KADJ T") - _safe(df, "B_KADJ T")
    feats["abs_tempo_diff"] = (_safe(df, "A_KADJ T") - _safe(df, "B_KADJ T")).abs()
    # pace classification
    for prefix in ["a", "b"]:
        col = "A_KADJ T" if prefix == "a" else "B_KADJ T"
        tempo = _safe(df, col)
        feats[f"{prefix}_pace_slow"] = (tempo < 65).astype(int)
        feats[f"{prefix}_pace_medium"] = ((tempo >= 65) & (tempo <= 70)).astype(int)
        feats[f"{prefix}_pace_fast"] = (tempo > 70).astype(int)

    feats["delta_ast_rate"] = _safe(df, "A_AST%") - _safe(df, "B_AST%")
    feats["delta_blk_rate"] = _safe(df, "A_BLK%") - _safe(df, "B_BLK%")

    a_toa = _safe(df, "A_TOV%") / _safe(df, "A_AST%").replace(0, np.nan)
    b_toa = _safe(df, "B_TOV%") / _safe(df, "B_AST%").replace(0, np.nan)
    feats["delta_toa_ratio"] = a_toa - b_toa

    # SOS and context
    feats["delta_elite_sos"] = _safe(df, "A_ELITE SOS") - _safe(df, "B_ELITE SOS")
    feats["delta_wab"] = _safe(df, "A_WAB") - _safe(df, "B_WAB")
    feats["delta_q1_wins"] = _safe(df, "A_Q1_W") - _safe(df, "B_Q1_W")
    feats["delta_q1q2_wins"] = _safe(df, "A_Q1Q2_W") - _safe(df, "B_Q1Q2_W")
    feats["delta_conf_avg_adjem"] = _safe(df, "A_CONF_AVG_ADJEM") - _safe(df, "B_CONF_AVG_ADJEM")
    feats["delta_resume"] = _safe(df, "A_RESUME") - _safe(df, "B_RESUME")
    feats["delta_elo"] = _safe(df, "A_ELO") - _safe(df, "B_ELO")

    # seed features
    feats["seed_a"] = df["SEED_A"].astype(float)
    feats["seed_b"] = df["SEED_B"].astype(float)
    feats["seed_diff"] = df["SEED_A"].astype(float) - df["SEED_B"].astype(float)
    feats["seed_ratio"] = df["SEED_A"].astype(float) / df["SEED_B"].astype(float).replace(0, np.nan)

    # historical program features
    feats["delta_hist_win_rate"] = _safe(df, "A_HIST_WIN_RATE") - _safe(df, "B_HIST_WIN_RATE")
    feats["delta_hist_s16_rate"] = _safe(df, "A_HIST_S16_RATE") - _safe(df, "B_HIST_S16_RATE")
    feats["delta_hist_f4_rate"] = _safe(df, "A_HIST_F4_RATE") - _safe(df, "B_HIST_F4_RATE")
    feats["a_blue_blood"] = _safe(df, "A_BLUE_BLOOD", 0).fillna(0)
    feats["b_blue_blood"] = _safe(df, "B_BLUE_BLOOD", 0).fillna(0)
    feats["delta_blue_blood"] = feats["a_blue_blood"] - feats["b_blue_blood"]

    # EvanMiya and Z ratings
    feats["delta_em_rating"] = _safe(df, "A_EM_RATING") - _safe(df, "B_EM_RATING")
    feats["delta_em_killshots"] = _safe(df, "A_EM_KILLSHOTS") - _safe(df, "B_EM_KILLSHOTS")
    feats["delta_z_rating"] = _safe(df, "A_Z_RATING") - _safe(df, "B_Z_RATING")

    # height and experience
    feats["delta_avg_hgt"] = _safe(df, "A_AVG HGT") - _safe(df, "B_AVG HGT")
    feats["delta_eff_hgt"] = _safe(df, "A_EFF HGT") - _safe(df, "B_EFF HGT")
    feats["delta_exp"] = _safe(df, "A_EXP") - _safe(df, "B_EXP")
    feats["delta_talent"] = _safe(df, "A_TALENT") - _safe(df, "B_TALENT")
    # composite features
    a_balanced = ((_safe(df, "A_KADJ O RANK") <= 30).astype(int) +
                  (_safe(df, "A_KADJ D RANK") <= 30).astype(int))
    b_balanced = ((_safe(df, "B_KADJ O RANK") <= 30).astype(int) +
                  (_safe(df, "B_KADJ D RANK") <= 30).astype(int))
    feats["a_balanced_profile"] = a_balanced
    feats["b_balanced_profile"] = b_balanced
    feats["delta_balanced_profile"] = a_balanced - b_balanced

    a_bart_ratio = _safe(df, "A_BADJ O") / _safe(df, "A_BADJ D").replace(0, np.nan)
    b_bart_ratio = _safe(df, "B_BADJ O") / _safe(df, "B_BADJ D").replace(0, np.nan)
    feats["delta_bart_eff_ratio"] = a_bart_ratio - b_bart_ratio

    a_trap = ((_safe(df, "A_KADJ EM") > 25) &
              (_safe(df, "A_KADJ T") >= 64) &
              (_safe(df, "A_KADJ T") <= 72)).astype(int)
    b_trap = ((_safe(df, "B_KADJ EM") > 25) &
              (_safe(df, "B_KADJ T") >= 64) &
              (_safe(df, "B_KADJ T") <= 72)).astype(int)
    feats["a_trapezoid"] = a_trap
    feats["b_trapezoid"] = b_trap
    feats["delta_trapezoid"] = a_trap - b_trap

    def _norm(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    a_readiness = (
        _norm(_safe(df, "A_ELITE SOS").fillna(0)) * 0.25 +
        _norm(_safe(df, "A_WAB").fillna(0)) * 0.25 +
        _norm(_safe(df, "A_Q1_W").fillna(0)) * 0.25 +
        _norm(_safe(df, "A_EXP").fillna(2)) * 0.25
    )
    b_readiness = (
        _norm(_safe(df, "B_ELITE SOS").fillna(0)) * 0.25 +
        _norm(_safe(df, "B_WAB").fillna(0)) * 0.25 +
        _norm(_safe(df, "B_Q1_W").fillna(0)) * 0.25 +
        _norm(_safe(df, "B_EXP").fillna(2)) * 0.25
    )
    feats["a_tourney_readiness"] = a_readiness
    feats["b_tourney_readiness"] = b_readiness
    feats["delta_tourney_readiness"] = a_readiness - b_readiness
    feats["tempo_mismatch"] = (_safe(df, "A_KADJ T") - _safe(df, "B_KADJ T")).abs()

    power_confs = {"SEC", "B12", "B10", "ACC", "BE", "P12"}
    a_upset_flag = (
        (~_safe(df, "A_CONF").isin(power_confs)) &
        (_safe(df, "A_KADJ EM RANK") <= 40) &
        (df["SEED_A"] >= 10)
    ).astype(int)
    b_upset_flag = (
        (~_safe(df, "B_CONF").isin(power_confs)) &
        (_safe(df, "B_KADJ EM RANK") <= 40) &
        (df["SEED_B"] >= 10)
    ).astype(int)
    feats["a_upset_propensity"] = a_upset_flag
    feats["b_upset_propensity"] = b_upset_flag

    feats["defense_offense_edge_a"] = (
        _safe(df, "B_EFG%") - _safe(df, "A_EFG%D")
    )
    feats["defense_offense_edge_b"] = (
        _safe(df, "A_EFG%") - _safe(df, "B_EFG%D")
    )
    feats["delta_badj_em_nt"] = _safe(df, "A_NT_BADJ_EM") - _safe(df, "B_NT_BADJ_EM")
    feats["delta_win_pct"] = _safe(df, "A_WIN%") - _safe(df, "B_WIN%")
    feats["a_win_pct"] = _safe(df, "A_WIN%")
    feats["b_win_pct"] = _safe(df, "B_WIN%")

    # --- upset-tuned features ---
    abs_em_gap = (_safe(df, "A_BADJ EM") - _safe(df, "B_BADJ EM")).abs()
    feats["abs_em_gap"] = abs_em_gap
    feats["is_tossup"] = (abs_em_gap < 5).astype(float)

    feats["delta_ps_em_change"] = _safe(df, "A_PS_EM_CHANGE") - _safe(df, "B_PS_EM_CHANGE")
    feats["delta_ps_rank_change"] = _safe(df, "A_PS_RANK_CHANGE") - _safe(df, "B_PS_RANK_CHANGE")
    a_peaking = (_safe(df, "A_PS_RANK_CHANGE") >= 10).astype(float)
    b_peaking = (_safe(df, "B_PS_RANK_CHANGE") >= 10).astype(float)
    feats["delta_peaking"] = a_peaking - b_peaking

    feats["delta_ap_rank_final"] = _safe(df, "A_AP_RANK_FINAL") - _safe(df, "B_AP_RANK_FINAL")
    feats["a_is_ranked"] = (_safe(df, "A_AP_RANK_FINAL") < 26).astype(float)
    feats["b_is_ranked"] = (_safe(df, "B_AP_RANK_FINAL") < 26).astype(float)
    feats["delta_ranked"] = feats["a_is_ranked"] - feats["b_is_ranked"]

    a_3pt_vol = _safe(df, "A_3PTR", 0) * (1.0 - _safe(df, "A_3PT%", 0.33))
    b_3pt_vol = _safe(df, "B_3PTR", 0) * (1.0 - _safe(df, "B_3PT%", 0.33))
    feats["a_3pt_volatility"] = a_3pt_vol
    feats["b_3pt_volatility"] = b_3pt_vol
    feats["delta_3pt_volatility"] = a_3pt_vol - b_3pt_vol

    avg_tempo = (_safe(df, "A_KADJ T") + _safe(df, "B_KADJ T")) / 2.0
    a_pace_ctrl = (_safe(df, "A_KADJ T") - avg_tempo).abs()
    b_pace_ctrl = (_safe(df, "B_KADJ T") - avg_tempo).abs()
    feats["pace_control_edge"] = b_pace_ctrl - a_pace_ctrl

    feats["delta_ft_clutch"] = _safe(df, "A_FT%") - _safe(df, "B_FT%")
    feats["delta_recent_hist"] = (
        _safe(df, "A_HIST_WIN_RATE") * _safe(df, "A_HIST_S16_RATE") -
        _safe(df, "B_HIST_WIN_RATE") * _safe(df, "B_HIST_S16_RATE")
    )

    return feats


def run_rfe(feature_matrix, target, cv_folds=5, n_jobs=-1, random_state=42):
    """
    run recursive feature elimination with cross-validation to find optimal feature subset.
    returns the selected feature names and the fitted RFECV object.
    """
    valid_mask = feature_matrix.notna().all(axis=1) & target.notna()
    X = feature_matrix[valid_mask].values
    y = target[valid_mask].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    estimator = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
    rfecv = RFECV(estimator=estimator, cv=cv_folds, scoring="neg_log_loss",
                  n_jobs=n_jobs, min_features_to_select=10)
    rfecv.fit(X_scaled, y)

    selected = feature_matrix.columns[rfecv.support_].tolist()
    print(f"RFE selected {len(selected)} features from {feature_matrix.shape[1]}")
    return selected, rfecv


def prepare_training_data(matchup_df):
    """
    build feature matrix and target from the labeled matchup dataset.
    returns X (DataFrame), y (Series), and the full feature column list.
    """
    feature_df = build_features(matchup_df)
    target = matchup_df["TEAM_A_WIN"].astype(int)
    feature_df = feature_df.reset_index(drop=True)
    target = target.reset_index(drop=True)
    return feature_df, target


def impute_features(feature_df, strategy="median"):
    """fill missing values using the specified strategy."""
    for col in feature_df.columns:
        if feature_df[col].isna().any():
            if strategy == "median":
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())
            else:
                feature_df[col] = feature_df[col].fillna(0)
    return feature_df


if __name__ == "__main__":
    from data_loader import build_matchup_dataset
    print("Building matchup dataset...")
    md = build_matchup_dataset()
    print("Building features...")
    X, y = prepare_training_data(md)
    X = impute_features(X)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print("Running RFE...")
    selected, rfecv = run_rfe(X, y)
    print(f"Selected features: {selected[:10]}...")
