#!/usr/bin/env python3
"""Phase H — FT-Transformer SOTA tabular DL baseline.

Issue 1 of the ISPRS-calibrated peer review asked for a competitive DL baseline
against XGBoost. FT-Transformer (Gorishniy et al. 2021, NeurIPS) is the SOTA
tabular DL architecture: it tokenises each feature into a learned embedding
and applies a small Transformer to the sequence of feature tokens plus a CLS
token; the CLS output feeds a regression head.

We tune via Optuna over 50 trials and report the best OOF spatial-CV RMSE.
Same dataset, same folds, same NaN-imputation as MLP/TabNet (median per
column). Direct comparison with XGBoost in Table T7.

Library: rtdl_revisiting_models (Gorishniy et al. 2024 update).
CPU-only (CUDA not available on this machine). ETA ~6-12h.
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
from rtdl_revisiting_models import FTTransformer

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
SAMP = ROOT / "scale_p1" / "samples_unified"
OUT = ROOT / "paper" / "experiments"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_TRIALS = 15  # reduced from 50 (CPU constraint: ~1-2h/trial → ~24h total)


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


def get_splits(df):
    t = pyproj.Transformer.from_crs(4326, 32719, always_xy=True)
    x_utm, y_utm = t.transform(df.lon.values, df.lat.values)
    blocks = (x_utm // 10000).astype(int) * 100000 + (y_utm // 10000).astype(int)
    gkf = GroupKFold(n_splits=5)
    return list(gkf.split(np.zeros(len(df)), df.residual_corrected.values,
                          groups=blocks))


def train_fold(X_tr, y_tr, X_te, params, n_features, max_epochs=30, patience=5):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sc = StandardScaler().fit(X_tr)
    Xtr_s = sc.transform(X_tr).astype(np.float32)
    Xte_s = sc.transform(X_te).astype(np.float32)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(Xtr_s))
    n_val = len(Xtr_s) // 5
    val, tri = idx[:n_val], idx[n_val:]

    model = FTTransformer(
        n_cont_features=n_features,
        cat_cardinalities=[],
        d_out=1,
        n_blocks=params["n_blocks"],
        d_block=params["d_block"],
        attention_n_heads=params["attention_n_heads"],
        attention_dropout=params["attention_dropout"],
        ffn_d_hidden=None,
        ffn_d_hidden_multiplier=params["ffn_d_hidden_multiplier"],
        ffn_dropout=params["ffn_dropout"],
        residual_dropout=0.0,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                             weight_decay=params["weight_decay"])
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(Xtr_s[tri]),
                       torch.from_numpy(y_tr[tri].astype(np.float32)))
    dl = DataLoader(ds, batch_size=params["batch"], shuffle=True)
    Xv = torch.from_numpy(Xtr_s[val]).to(DEVICE)
    yv = y_tr[val]
    Xt = torch.from_numpy(Xte_s).to(DEVICE)

    best_val, best_state, p_ctr = np.inf, None, 0
    for ep in range(max_epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb, x_cat=None).squeeze(-1)
            loss_fn(pred, yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v_pred = model(Xv, x_cat=None).squeeze(-1).cpu().numpy()
            v_rmse = float(np.sqrt(mean_squared_error(yv, v_pred)))
        if v_rmse < best_val - 1e-4:
            best_val = v_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            p_ctr = 0
        else:
            p_ctr += 1
            if p_ctr >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(Xt, x_cat=None).squeeze(-1).cpu().numpy()


def objective(trial, df, FEATURES, splits):
    # Reduced search space to keep per-trial time tractable on CPU. d_block
    # capped at 128, batch ≥ 2048 (larger batches train faster per epoch on
    # this CPU). Keep architecture diversity via n_blocks and n_heads.
    params = {
        "n_blocks": trial.suggest_int("n_blocks", 2, 3),
        "d_block": trial.suggest_categorical("d_block", [64, 128]),
        "attention_n_heads": trial.suggest_categorical("attention_n_heads", [4, 8]),
        "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.3),
        "ffn_d_hidden_multiplier": 4.0 / 3.0,  # locked: rtdl default
        "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.3),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch": trial.suggest_categorical("batch", [2048, 4096]),
    }
    # d_block must be divisible by n_heads
    if params["d_block"] % params["attention_n_heads"] != 0:
        return float("inf")

    X = df[FEATURES].values
    y = df.residual_corrected.values
    preds = np.full_like(y, np.nan, dtype=np.float64)
    for tr, te in splits:
        try:
            preds[te] = train_fold(X[tr], y[tr], X[te], params, len(FEATURES))
        except Exception as e:
            print(f"  trial failed: {type(e).__name__}: {e}")
            return float("inf")
    return float(np.sqrt(mean_squared_error(y, preds)))


def main():
    t_total = time.time()
    print("=" * 70)
    print(f"Phase H — FT-Transformer tabular DL baseline  (device={DEVICE})")
    print(f"  N_TRIALS = {N_TRIALS}")
    print("=" * 70)
    df, FEATURES = load_data()
    print(f"Loaded {len(df)} rows, {len(FEATURES)} features")
    splits = get_splits(df)

    def log_trial(study, trial):
        if trial.value != float("inf"):
            print(f"  trial {trial.number+1:>2d}/{N_TRIALS}  "
                  f"RMSE={trial.value:.4f}  "
                  f"best={study.best_value:.4f}  "
                  f"elapsed={(time.time()-t_total)/60:.1f}m", flush=True)

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: objective(t, df, FEATURES, splits),
                   n_trials=N_TRIALS, show_progress_bar=False,
                   callbacks=[log_trial])
    print(f"\n[FT-Transformer] best OOF RMSE: {study.best_value:.4f}")
    print(f"  best params: {study.best_params}")

    XGB_RMSE = 2.483180756453067
    delta_pct = 100 * (study.best_value - XGB_RMSE) / XGB_RMSE
    print(f"\n=== Comparison vs XGBoost ===")
    print(f"  XGBoost     : 2.4832 m  (-18.7% vs raw)")
    print(f"  FT-Transformer: {study.best_value:.4f} m  ({delta_pct:+.2f}% vs XGBoost)")

    out = {
        "best_rmse": study.best_value,
        "best_params": study.best_params,
        "xgboost_ref": XGB_RMSE,
        "delta_pct_vs_xgboost": delta_pct,
        "n_trials": N_TRIALS,
        "wall_time_h": (time.time() - t_total) / 3600,
    }
    (OUT / "phase_H_ft_transformer.json").write_text(json.dumps(out, indent=2))
    print(f"\n→ {OUT / 'phase_H_ft_transformer.json'}")
    print(f"Total wall time: {out['wall_time_h']:.1f} h")


if __name__ == "__main__":
    main()
