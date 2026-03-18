import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "predictions")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# held-out evaluation years
EVAL_YEARS = [2019, 2022, 2023]
TRAIN_YEAR_MAX = 2023


def build_pipelines(random_state=42):
    """construct all model pipelines."""
    pipelines = {}

    # logistic regression baseline
    pipelines["logistic"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=0.5, random_state=random_state)),
    ])

    # ridge classifier (primary - best historical record for champion prediction)
    # calibrate to get probability outputs
    ridge_base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RidgeClassifier(alpha=1.0)),
    ])
    pipelines["ridge"] = CalibratedClassifierCV(ridge_base, cv=5, method="sigmoid")

    # random forest - best for feature importance
    pipelines["random_forest"] = RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=5,
        max_features="sqrt", random_state=random_state, n_jobs=-1
    )

    # gradient boosting
    if HAS_LGB:
        pipelines["lgbm"] = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=31, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.8, random_state=random_state, verbose=-1
        )
    elif HAS_XGB:
        pipelines["xgboost"] = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=random_state, use_label_encoder=False
        )

    return pipelines


def train_and_evaluate(X, y, years, selected_features=None, random_state=42):
    """
    train all models on years <= TRAIN_YEAR_MAX and evaluate on EVAL_YEARS.
    returns dict of {model_name: {"model": fitted, "cv_logloss": float, "eval_results": dict}}
    """
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if selected_features is not None:
        X = X[selected_features]

    # impute any remaining NaNs
    X = X.copy().fillna(X.median())

    pipelines = build_pipelines(random_state)
    results = {}

    train_mask = years <= TRAIN_YEAR_MAX
    X_train = X[train_mask]
    y_train = y[train_mask]

    print(f"Training on {train_mask.sum()} games, evaluating on {(~train_mask).sum()} games")

    for name, model in pipelines.items():
        print(f"\nTraining {name}...")

        # cross-validation on train set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring="neg_log_loss", n_jobs=-1)
        cv_logloss = -cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"  CV log-loss: {cv_logloss:.4f} +/- {cv_std:.4f}")

        # fit on full training set
        model.fit(X_train, y_train)

        # evaluate on held-out years
        eval_results = {}
        for yr in EVAL_YEARS:
            yr_mask = years == yr
            if yr_mask.sum() == 0:
                continue
            X_eval = X[yr_mask].fillna(X_train.median())
            y_eval = y[yr_mask]

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_eval)[:, 1]
            else:
                y_prob = model.decision_function(X_eval)
                y_prob = 1 / (1 + np.exp(-y_prob))

            y_pred = (y_prob >= 0.5).astype(int)
            eval_logloss = log_loss(y_eval, y_prob)
            eval_acc = accuracy_score(y_eval, y_pred)
            eval_brier = brier_score_loss(y_eval, y_prob)
            eval_results[yr] = {
                "log_loss": round(eval_logloss, 4),
                "accuracy": round(eval_acc, 4),
                "brier_score": round(eval_brier, 4),
                "n_games": int(yr_mask.sum()),
            }
            print(f"  {yr}: log-loss={eval_logloss:.4f} acc={eval_acc:.4f}")

        results[name] = {
            "model": model,
            "cv_logloss": round(cv_logloss, 4),
            "cv_std": round(cv_std, 4),
            "eval_results": eval_results,
        }

        # save model to disk
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))

    return results


def build_ensemble(results, X_train, y_train, selected_features=None, random_state=42):
    """
    build weighted ensemble of ridge and random_forest using CV-tuned weights.
    returns a simple wrapper with predict_proba.
    """
    if selected_features is not None:
        X_train = X_train[selected_features]
    X_train = X_train.fillna(X_train.median())

    ridge = results.get("ridge", {}).get("model")
    rf = results.get("random_forest", {}).get("model")

    if ridge is None or rf is None:
        return None

    # tune ensemble weights by CV log-loss on the training set
    best_w, best_ll = 0.5, np.inf
    for w in np.linspace(0.1, 0.9, 17):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        lls = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            Xtr, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            ytr, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            ridge.fit(Xtr, ytr)
            rf.fit(Xtr, ytr)
            p_ridge = ridge.predict_proba(Xv)[:, 1]
            p_rf = rf.predict_proba(Xv)[:, 1]
            p_ens = w * p_ridge + (1 - w) * p_rf
            lls.append(log_loss(yv, p_ens))
        mean_ll = np.mean(lls)
        if mean_ll < best_ll:
            best_ll, best_w = mean_ll, w

    print(f"Ensemble: best ridge weight={best_w:.2f}, CV log-loss={best_ll:.4f}")

    # refit both on full training data
    ridge.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    class EnsembleModel:
        def __init__(self, ridge, rf, w):
            self.ridge = ridge
            self.rf = rf
            self.w = w

        def predict_proba(self, X):
            X = X.fillna(0)
            p_r = self.ridge.predict_proba(X)[:, 1]
            p_f = self.rf.predict_proba(X)[:, 1]
            p = self.w * p_r + (1 - self.w) * p_f
            return np.column_stack([1 - p, p])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    ens = EnsembleModel(ridge, rf, best_w)
    joblib.dump(ens, os.path.join(MODELS_DIR, "ensemble.joblib"))
    return ens, best_w


def plot_feature_importance(model, feature_names, top_n=30, year=2026):
    """save a feature importance bar chart from the random forest model."""
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices][::-1], align="center")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=8)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Top {top_n} Feature Importances - Random Forest")
    plt.tight_layout()

    out_path = os.path.join(PREDICTIONS_DIR, f"feature_importance_{year}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Feature importance chart saved to {out_path}")
    return out_path


def save_eval_summary(results, path=None):
    """save model evaluation summary to JSON."""
    if path is None:
        path = os.path.join(PREDICTIONS_DIR, "model_eval_summary.json")
    summary = {}
    for name, res in results.items():
        summary[name] = {
            "cv_logloss": res["cv_logloss"],
            "cv_std": res.get("cv_std"),
            "eval_results": res["eval_results"],
        }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Eval summary saved to {path}")
    return summary


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import build_matchup_dataset
    from feature_engineering import prepare_training_data, impute_features, run_rfe

    print("Loading matchup data...")
    md = build_matchup_dataset()
    X, y = prepare_training_data(md)
    X = impute_features(X)
    years = md["YEAR"].reset_index(drop=True)

    print("Running RFE to select features...")
    train_mask = years <= TRAIN_YEAR_MAX
    selected, rfecv = run_rfe(X[train_mask], y[train_mask])

    print("Training all models...")
    results = train_and_evaluate(X, y, years, selected_features=selected)

    save_eval_summary(results)

    rf_model = results.get("random_forest", {}).get("model")
    if rf_model is not None:
        plot_feature_importance(rf_model, selected if selected else list(X.columns))

    print("Building ensemble...")
    train_mask_bool = years <= TRAIN_YEAR_MAX
    ens, ens_w = build_ensemble(results, X[train_mask_bool], y[train_mask_bool], selected_features=selected)
    print("Done.")
