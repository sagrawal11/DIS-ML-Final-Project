"""Modelling helpers for notebook 02.

Three models of increasing complexity:
  * PhysicsBaseline  - linear regression, per-compound tyre-age slope + fuel proxy
                       (a Heilmeier-style lap-time decomposition: tyre wear + fuel burn).
  * XGBoost          - gradient boosting on the full encoded feature matrix.
  * ResidualModel    - XGBoost trained on the physics baseline's residuals; the
                       hypothesis is that physical relationships transfer across
                       the 2026 regulation change better than raw patterns.

Plus metrics (MAE/RMSE in seconds) and a per-group breakdown that flags unseen
entities (e.g. the Cadillac team and the Madrid circuit in 2026).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

SEED = 42
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# XGBoost defaults; tuned values override these (see notebook 04).
XGB_DEFAULTS = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
                    "tree_method": "hist", "random_state": SEED, "n_jobs": -1,
                    "objective": "reg:squarederror"}


def make_xgb(**overrides):
    return XGBRegressor(**{**XGB_DEFAULTS, **overrides})


def metrics(y, yhat):
    """MAE and RMSE in the target's units (seconds)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return {"MAE": float(mean_absolute_error(y, yhat)),
            "RMSE": float(np.sqrt(mean_squared_error(y, yhat)))}


class PhysicsBaseline:
    """Linear regression with a per-compound intercept and tyre-age slope, plus a
    single fuel-proxy slope. No global intercept (each compound carries its own)."""

    def __init__(self, compounds=DRY_COMPOUNDS):
        self.compounds = compounds

    def _design(self, df):
        X = pd.DataFrame(index=df.index)
        for c in self.compounds:
            on = df.compound == c
            X[f"cmp_{c}"] = on.astype(float)          # per-compound intercept
            X[f"age_{c}"] = df.tyre_age.where(on, 0.0).astype(float)  # per-compound slope
        X["fuel_proxy"] = df.fuel_proxy.astype(float)
        return X

    def fit(self, df, y):
        self.model = LinearRegression(fit_intercept=False).fit(self._design(df), y)
        return self

    def predict(self, df):
        return self.model.predict(self._design(df))


class ResidualModel:
    """Physics baseline + XGBoost fit on its residuals. prediction = physics + residual.

    X is the pre-encoded feature matrix (aligned columns across all splits); df
    supplies the raw columns the physics baseline needs.
    """

    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.phys = PhysicsBaseline()

    def fit(self, df, X, y):
        self.phys.fit(df, y)
        resid = np.asarray(y, float) - self.phys.predict(df)
        self.xgb = make_xgb(**self.xgb_params).fit(X, resid)
        return self

    def predict(self, df, X):
        return self.phys.predict(df) + self.xgb.predict(X)


MODELS = ["Physics", "XGBoost", "Residual"]
COLORS = {"Physics": "#718096", "XGBoost": "#2b6cb0", "Residual": "#c53030"}
PARAMS_PATH = "results/xgb_params.json"   # written by notebook 04, read by 05-07


def fit_predict(name, df, X, y, tr, te, params):
    """Fit model `name` on the rows in mask `tr` and predict the rows in mask `te`.

    df holds the raw columns (the physics baseline needs them), X the aligned
    encoded feature matrix, y the target; params are the tuned XGBoost params.
    """
    dtr, Xtr, ytr = df[tr], X[tr], np.asarray(y[tr], float)
    dte, Xte = df[te], X[te]
    if name == "Physics":
        return PhysicsBaseline().fit(dtr, ytr).predict(dte)
    if name == "XGBoost":
        return make_xgb(**params).fit(Xtr, ytr).predict(Xte)
    if name == "Residual":
        return ResidualModel(**params).fit(dtr, Xtr, ytr).predict(dte, Xte)
    raise ValueError(name)


def evaluate_split(df, X, y, tr, te, params, models=MODELS):
    """MAE/RMSE of every model for one train/test split, one row per model."""
    yte = np.asarray(y[te], float)
    return pd.DataFrame({n: metrics(yte, fit_predict(n, df, X, y, tr, te, params))
                         for n in models}).T


def breakdown(df, y, yhat, by, unseen=None):
    """Per-group MAE/RMSE, sorted worst-first. `unseen` flags group values that did
    not appear in the training era (e.g. {'Cadillac'} or {'Madrid'})."""
    g = pd.DataFrame({by: np.asarray(df[by]), "y": np.asarray(y, float),
                      "yhat": np.asarray(yhat, float)})
    rows = []
    for name, sub in g.groupby(by):
        m = metrics(sub.y, sub.yhat)
        rows.append({by: name, "n": len(sub), "MAE": round(m["MAE"], 3),
                     "RMSE": round(m["RMSE"], 3), "unseen": bool(unseen and name in unseen)})
    return pd.DataFrame(rows).sort_values("MAE", ascending=False).reset_index(drop=True)


def _selfcheck():
    rng = np.random.RandomState(SEED)
    n = 800
    df = pd.DataFrame({
        "compound": rng.choice(DRY_COMPOUNDS, n),
        "tyre_age": rng.randint(0, 30, n).astype(float),
        "fuel_proxy": rng.randint(0, 50, n).astype(float),
    })
    slope = {"SOFT": 0.08, "MEDIUM": 0.05, "HARD": 0.03}
    base = {"SOFT": -0.3, "MEDIUM": 0.0, "HARD": 0.3}
    y = (df.compound.map(base) + df.tyre_age * df.compound.map(slope)
         + 0.02 * df.fuel_proxy + rng.normal(0, 0.05, n)).values

    pb = PhysicsBaseline().fit(df, y)
    coef = dict(zip(pb._design(df).columns, pb.model.coef_))
    assert abs(coef["age_SOFT"] - 0.08) < 0.02, coef          # recovers per-compound slope
    assert metrics(y, pb.predict(df))["MAE"] < 0.1

    # Residual XGBoost should beat physics-only on a target with a nonlinear wiggle.
    X = pd.get_dummies(df[["compound"]], dtype=int).astype(float)
    X["tyre_age"], X["fuel_proxy"] = df.tyre_age, df.fuel_proxy
    y2 = y + 0.6 * np.sin(df.tyre_age / 3.0).values
    rm = ResidualModel(n_estimators=60, max_depth=3).fit(df, X, y2)
    phys_only = PhysicsBaseline().fit(df, y2)
    assert metrics(y2, rm.predict(df, X))["MAE"] < metrics(y2, phys_only.predict(df))["MAE"]

    # fit_predict / evaluate_split route to the same models on boolean masks.
    tr = pd.Series(np.arange(n) < 600); te = ~tr
    ev = evaluate_split(df, X, pd.Series(y2), tr, te, {"n_estimators": 60, "max_depth": 3})
    assert list(ev.index) == MODELS and ev.loc["Residual", "MAE"] < ev.loc["Physics", "MAE"]

    bd = breakdown(df, y, pb.predict(df), "compound", unseen={"SOFT"})
    assert bool(bd.loc[bd.compound == "SOFT", "unseen"].iloc[0]) is True
    print("models.py self-check passed")


if __name__ == "__main__":
    _selfcheck()
