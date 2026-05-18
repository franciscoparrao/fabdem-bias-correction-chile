#!/usr/bin/env python3
"""Deep-learning tabular baselines vs XGBoost (same spatial CV).

Models:
  - MLP (3 hidden layers, 256 units, ReLU + dropout)
  - TabNet (pytorch_tabnet defaults, but matched epochs/early-stopping)

Uses the same 10-km spatial GroupKFold (K=5) as XGBoost so RMSE is directly
comparable to the locked-in 2.483 m.

Output: dl_baseline_results.json
"""
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pyproj
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
SAMP = ROOT / "scale_p1" / "samples_unified"
OUT_JSON = ROOT / "paper" / "experiments" / "dl_baseline_results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

GROUPS = {
    "terrain_1st": ["slope", "aspect", "hillshade", "eastness", "northness"],
    "terrain_2nd": ["curvature", "tpi", "tri", "vrm", "dev", "convergence"],
    "shape_3d": ["openness_negative", "openness_positive", "svf"],
    "landform": ["mrvbf", "mrrtf", "geomorphons"],
    "hydro": ["hand", "twi", "flow_accumulation", "flow_accumulation_mfd",
              "flow_direction_d8", "flow_direction_dinf", "stream_network", "valley_depth"],
    "s2_optical": ["ndvi", "ndwi", "ndmi", "bsi", "ndbi"],
    "s1_sar": ["s1_vh_db", "s1_vv_db", "s1_vv_vh_ratio"],
}
ALL_FEATURES = [f for g in GROUPS.values() for f in g]


def load_data():
    df = pd.read_csv(SAMP / "samples_mm_full.csv")
    if "stream_network" in df.columns:
        df["stream_network"] = df["stream_network"].fillna(0)
    # DL needs no NaNs; impute median per column
    for c in ALL_FEATURES:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    keep = np.isfinite(df["residual_corrected"].values)
    df = df.loc[keep].reset_index(drop=True)
    return df


def get_blocks_10km(lon, lat):
    t = pyproj.Transformer.from_crs(4326, 32719, always_xy=True)
    x_utm, y_utm = t.transform(lon, lat)
    block_x = (x_utm // 10000.0).astype(int)
    block_y = (y_utm // 10000.0).astype(int)
    return block_x * 100000 + block_y


# ---------------------- MLP ----------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_fold(X_tr, y_tr, X_te, y_te, max_epochs=200, patience=20, batch=2048):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    # Internal val split 80/20 from train (random, fold already spatial)
    n = len(X_tr_s)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    n_val = n // 5
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    ds_tr = TensorDataset(
        torch.from_numpy(X_tr_s[tr_idx]),
        torch.from_numpy(y_tr[tr_idx].astype(np.float32)))
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True)
    X_val_t = torch.from_numpy(X_tr_s[val_idx]).to(DEVICE)
    y_val_t = torch.from_numpy(y_tr[val_idx].astype(np.float32)).to(DEVICE)
    X_te_t = torch.from_numpy(X_te_s).to(DEVICE)

    model = MLP(X_tr_s.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = np.inf
    best_state = None
    patience_ctr = 0
    for ep in range(max_epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v_pred = model(X_val_t).cpu().numpy()
            v_rmse = float(np.sqrt(mean_squared_error(y_tr[val_idx], v_pred)))
        if v_rmse < best_val - 1e-4:
            best_val = v_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_pred = model(X_te_t).cpu().numpy()
    return te_pred, ep + 1


def run_mlp(df, splits):
    print(f"\n[MLP] device={DEVICE}")
    X = df[ALL_FEATURES].values
    y = df["residual_corrected"].values
    preds = np.full_like(y, np.nan, dtype=np.float64)
    per_fold = []
    for fi, (tr, te) in enumerate(splits):
        t0 = time.time()
        te_pred, n_ep = train_mlp_fold(X[tr], y[tr], X[te], y[te])
        preds[te] = te_pred
        rmse = float(np.sqrt(mean_squared_error(y[te], te_pred)))
        per_fold.append({"fold": fi, "n_test": int(te.size),
                         "rmse": rmse, "epochs": n_ep,
                         "elapsed_s": time.time() - t0})
        print(f"  fold {fi}: n={te.size:>5d} | RMSE={rmse:.4f} | "
              f"ep={n_ep:>3d} | {per_fold[-1]['elapsed_s']:.1f}s")
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    return {"rmse_oof": rmse, "mae_oof": mae, "folds": per_fold}


# ---------------------- TabNet ----------------------
def run_tabnet(df, splits):
    from pytorch_tabnet.tab_model import TabNetRegressor
    print(f"\n[TabNet] device={DEVICE}")
    X = df[ALL_FEATURES].values
    y = df["residual_corrected"].values.reshape(-1, 1)
    preds = np.full(len(df), np.nan, dtype=np.float64)
    per_fold = []
    for fi, (tr, te) in enumerate(splits):
        t0 = time.time()
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        sc = StandardScaler().fit(X[tr])
        Xtr_s = sc.transform(X[tr]).astype(np.float32)
        Xte_s = sc.transform(X[te]).astype(np.float32)
        # Internal val split for early stopping
        n = len(Xtr_s)
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(n)
        n_val = n // 5
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        model = TabNetRegressor(
            n_d=32, n_a=32, n_steps=4, gamma=1.3,
            seed=SEED, device_name=DEVICE,
            verbose=0,
            optimizer_params=dict(lr=2e-2),
        )
        model.fit(
            X_train=Xtr_s[tr_idx], y_train=y[tr][tr_idx],
            eval_set=[(Xtr_s[val_idx], y[tr][val_idx])],
            max_epochs=200, patience=20, batch_size=4096, virtual_batch_size=512,
            num_workers=0, drop_last=False,
        )
        te_pred = model.predict(Xte_s).flatten()
        preds[te] = te_pred
        rmse = float(np.sqrt(mean_squared_error(y[te].flatten(), te_pred)))
        per_fold.append({"fold": fi, "n_test": int(te.size),
                         "rmse": rmse, "elapsed_s": time.time() - t0})
        print(f"  fold {fi}: n={te.size:>5d} | RMSE={rmse:.4f} | "
              f"{per_fold[-1]['elapsed_s']:.1f}s")
    y_flat = y.flatten()
    rmse = float(np.sqrt(mean_squared_error(y_flat, preds)))
    mae = float(mean_absolute_error(y_flat, preds))
    return {"rmse_oof": rmse, "mae_oof": mae, "folds": per_fold}


def main():
    df = load_data()
    y = df["residual_corrected"].values
    lon = df["lon"].values
    lat = df["lat"].values
    rmse_base = float(np.sqrt(mean_squared_error(y, np.zeros_like(y))))
    print(f"FABDEM raw baseline: RMSE={rmse_base:.3f}")
    print(f"XGBoost ref (mm_metrics.json): RMSE=2.483")

    blocks = get_blocks_10km(lon, lat)
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.zeros(len(df)), y, groups=blocks))
    print(f"Spatial blocks 10 km: {len(np.unique(blocks))}  |  folds: 5")

    results = {
        "n_samples": len(df),
        "n_features": len(ALL_FEATURES),
        "baseline_rmse": rmse_base,
        "xgboost_ref_rmse": 2.483180756453067,
        "models": {},
    }

    # ---- MLP ----
    t0 = time.time()
    results["models"]["mlp"] = run_mlp(df, splits)
    results["models"]["mlp"]["total_elapsed_s"] = time.time() - t0
    print(f"  MLP OOF RMSE: {results['models']['mlp']['rmse_oof']:.4f}")

    # ---- TabNet ----
    t0 = time.time()
    try:
        results["models"]["tabnet"] = run_tabnet(df, splits)
        results["models"]["tabnet"]["total_elapsed_s"] = time.time() - t0
        print(f"  TabNet OOF RMSE: {results['models']['tabnet']['rmse_oof']:.4f}")
    except Exception as e:
        print(f"  TabNet FAILED: {type(e).__name__}: {e}")
        results["models"]["tabnet"] = {"error": f"{type(e).__name__}: {e}"}

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_JSON}")


if __name__ == "__main__":
    main()
