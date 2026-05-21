#!/usr/bin/env python3
"""Phase D — Fair DL baselines: Optuna 100 trials for MLP and TabNet.

Same spatial-CV folds as XGBoost (10 km blocks, K=5). Same 33-feature stack.
For each model, an Optuna TPE sampler searches architecture and optimisation
hyperparameters for 100 trials. The result is the best-of-100 OOF RMSE per
model, directly comparable to the XGBoost result obtained under the same
compute budget.

This script is CPU-only (CUDA not available on this machine). Expect ~10-20h
total wall time depending on early-stopping convergence.

Outputs: phase_D_dl_tuning.json
"""
import os
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pyproj
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
SAMP = ROOT / "scale_p1" / "samples_unified"
OUT = ROOT / "paper" / "experiments"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_TRIALS = 100


def load_data():
    df = pd.read_csv(SAMP / "samples_mm_full.csv")
    if "stream_network" in df.columns:
        df["stream_network"] = df["stream_network"].fillna(0)
    META = {"granule", "beam", "lon", "lat", "h_te", "h_te_unc",
            "night", "terrain_flg", "n_te_phot", "snow", "water", "cloud",
            "h_te_orthometric", "geoid_N", "residual_corrected",
            "fabdem", "filled", "tile"}
    FEATURES = [c for c in df.columns if c not in META and df[c].dtype != "O"]
    for c in FEATURES:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    keep = np.isfinite(df["residual_corrected"].values)
    df = df.loc[keep].reset_index(drop=True)
    return df, FEATURES


# ============================================================
# MLP
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, dropout):
        super().__init__()
        layers = []
        prev = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_fold(X_tr, y_tr, X_te, params, max_epochs=200, patience=20):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sc = StandardScaler().fit(X_tr)
    Xtr_s = sc.transform(X_tr).astype(np.float32)
    Xte_s = sc.transform(X_te).astype(np.float32)
    # 80/20 val from tr fold
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(Xtr_s))
    n_val = len(Xtr_s) // 5
    val, tri = idx[:n_val], idx[n_val:]
    ds = TensorDataset(torch.from_numpy(Xtr_s[tri]),
                       torch.from_numpy(y_tr[tri].astype(np.float32)))
    dl = DataLoader(ds, batch_size=params["batch"], shuffle=True)
    Xv = torch.from_numpy(Xtr_s[val]).to(DEVICE)
    yv = torch.from_numpy(y_tr[val].astype(np.float32)).to(DEVICE)
    Xt = torch.from_numpy(Xte_s).to(DEVICE)

    model = MLP(Xtr_s.shape[1], params["hidden"], params["n_layers"],
                params["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                             weight_decay=params["weight_decay"])
    loss_fn = nn.MSELoss()
    best_val, best_state, p_ctr = np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v_pred = model(Xv).cpu().numpy()
            v_rmse = float(np.sqrt(mean_squared_error(y_tr[val], v_pred)))
        if v_rmse < best_val - 1e-4:
            best_val, best_state, p_ctr = v_rmse, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            p_ctr += 1
            if p_ctr >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(Xt).cpu().numpy()


def mlp_objective(trial, df, FEATURES, splits):
    params = {
        "hidden": trial.suggest_categorical("hidden", [64, 128, 256, 512]),
        "n_layers": trial.suggest_int("n_layers", 2, 5),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch": trial.suggest_categorical("batch", [512, 1024, 2048, 4096]),
    }
    X = df[FEATURES].values
    y = df.residual_corrected.values
    preds = np.full_like(y, np.nan, dtype=np.float64)
    for tr, te in splits:
        try:
            preds[te] = train_mlp_fold(X[tr], y[tr], X[te], params)
        except Exception:
            return float("inf")
    return float(np.sqrt(mean_squared_error(y, preds)))


# ============================================================
# TabNet
# ============================================================
def train_tabnet_fold(X_tr, y_tr, X_te, params):
    from pytorch_tabnet.tab_model import TabNetRegressor
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sc = StandardScaler().fit(X_tr)
    Xtr_s = sc.transform(X_tr).astype(np.float32)
    Xte_s = sc.transform(X_te).astype(np.float32)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(Xtr_s))
    n_val = len(Xtr_s) // 5
    val, tri = idx[:n_val], idx[n_val:]
    model = TabNetRegressor(
        n_d=params["n_d"], n_a=params["n_a"], n_steps=params["n_steps"],
        gamma=params["gamma"], seed=SEED, device_name=DEVICE, verbose=0,
        optimizer_params=dict(lr=params["lr"]),
    )
    model.fit(
        X_train=Xtr_s[tri], y_train=y_tr[tri].reshape(-1, 1),
        eval_set=[(Xtr_s[val], y_tr[val].reshape(-1, 1))],
        max_epochs=150, patience=20, batch_size=params["batch_size"],
        virtual_batch_size=min(512, params["batch_size"]), num_workers=0,
        drop_last=False,
    )
    return model.predict(Xte_s).flatten()


def tabnet_objective(trial, df, FEATURES, splits):
    params = {
        "n_d": trial.suggest_categorical("n_d", [16, 32, 64]),
        "n_a": trial.suggest_categorical("n_a", [16, 32, 64]),
        "n_steps": trial.suggest_int("n_steps", 3, 7),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lr": trial.suggest_float("lr", 5e-3, 5e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [2048, 4096, 8192]),
    }
    X = df[FEATURES].values
    y = df.residual_corrected.values
    preds = np.full_like(y, np.nan, dtype=np.float64)
    for tr, te in splits:
        try:
            preds[te] = train_tabnet_fold(X[tr], y[tr], X[te], params)
        except Exception:
            return float("inf")
    return float(np.sqrt(mean_squared_error(y, preds)))


# ============================================================
# MAIN
# ============================================================
def main():
    t_total = time.time()
    print("=" * 70)
    print(f"Phase D — fair DL baselines  (device={DEVICE})")
    print(f"  N_TRIALS per model = {N_TRIALS}")
    print("=" * 70)
    df, FEATURES = load_data()
    print(f"Loaded {len(df)} rows, {len(FEATURES)} features")
    t = pyproj.Transformer.from_crs(4326, 32719, always_xy=True)
    x_utm, y_utm = t.transform(df.lon.values, df.lat.values)
    blocks = (x_utm // 10000).astype(int) * 100000 + (y_utm // 10000).astype(int)
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.zeros(len(df)), df.residual_corrected.values, groups=blocks))

    results = {"baseline_xgboost": 2.483180756453067, "n_trials": N_TRIALS}

    # -- MLP --
    print(f"\n[MLP] tuning over {N_TRIALS} trials")
    t0 = time.time()
    study_mlp = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_mlp.optimize(lambda t: mlp_objective(t, df, FEATURES, splits),
                       n_trials=N_TRIALS, show_progress_bar=False)
    dt = time.time() - t0
    print(f"  MLP best OOF RMSE: {study_mlp.best_value:.4f}  ({dt/60:.1f} min)")
    print(f"  MLP best params: {study_mlp.best_params}")
    results["mlp"] = {
        "best_rmse": study_mlp.best_value,
        "best_params": study_mlp.best_params,
        "elapsed_min": dt / 60,
    }
    (OUT / "phase_D_dl_tuning.json").write_text(json.dumps(results, indent=2))

    # -- TabNet --
    print(f"\n[TabNet] tuning over {N_TRIALS} trials")
    t0 = time.time()
    study_tn = optuna.create_study(direction="minimize",
                                   sampler=optuna.samplers.TPESampler(seed=42))
    study_tn.optimize(lambda t: tabnet_objective(t, df, FEATURES, splits),
                      n_trials=N_TRIALS, show_progress_bar=False)
    dt = time.time() - t0
    print(f"  TabNet best OOF RMSE: {study_tn.best_value:.4f}  ({dt/60:.1f} min)")
    print(f"  TabNet best params: {study_tn.best_params}")
    results["tabnet"] = {
        "best_rmse": study_tn.best_value,
        "best_params": study_tn.best_params,
        "elapsed_min": dt / 60,
    }
    results["wall_time_h"] = (time.time() - t_total) / 3600
    (OUT / "phase_D_dl_tuning.json").write_text(json.dumps(results, indent=2))
    print(f"\nTotal wall time: {results['wall_time_h']:.1f} h")
    print(f"→ {OUT / 'phase_D_dl_tuning.json'}")


if __name__ == "__main__":
    main()
