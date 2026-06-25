from pathlib import Path
import sys
import os
import copy
import itertools
import random
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path("/content/drive/MyDrive/Aueb_Thesis/Empirical_Study/DSSSM")
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(SRC_DIR))

from DSSSMCode import (
    DSSSM,
    train_dsssm_model,
    extract_dsssm_regimes,
    regime_diagnostics,
    daily_segment_summary,
)


def set_seed(seed=3):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available; using CPU.")
    print("Using device:", device)
    return device


def load_arrays():
    X_train = np.load(DATA_DIR / "X_train_4.npy")
    X_val = np.load(DATA_DIR / "X_val_4.npy")
    X_test = np.load(DATA_DIR / "X_test_4.npy")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    assert X_train.ndim == 3 and X_val.ndim == 3 and X_test.ndim == 3
    return X_train, X_val, X_test


def to_dsssm_tensors(X_train, X_val, X_test, device):
    # Input arrays: [days, minutes, features]
    # DSSSM tensors: [minutes, days, features]
    trainX = torch.tensor(np.transpose(X_train, (1, 0, 2)), dtype=torch.float32, device=device)
    validX = torch.tensor(np.transpose(X_val, (1, 0, 2)), dtype=torch.float32, device=device)
    testX = torch.tensor(np.transpose(X_test, (1, 0, 2)), dtype=torch.float32, device=device)

    # Reconstruction/regime-identification setup.
    trainY = trainX.clone()
    validY = validX.clone()
    testY = testX.clone()

    print("trainX:", trainX.shape, trainX.device)
    print("validX:", validX.shape, validX.device)
    print("testX:", testX.shape, testX.device)
    return trainX, trainY, validX, validY, testX, testY


def build_candidate_configs():
    search_space_dsssm = {
        "d_dim": [2],
        "z_dim": [2],
        "h_dim": [16],
        "lamda_b": [0.0, 1.0, 5.0, 10.0],
        "learning_rate": [1e-3],
    }
    candidate_configs = [
        {
            "d_dim": d_dim,
            "z_dim": z_dim,
            "h_dim": h_dim,
            "lamda_b": lamda_b,
            "learning_rate": learning_rate,
        }
        for d_dim, z_dim, h_dim, lamda_b, learning_rate in itertools.product(
            search_space_dsssm["d_dim"],
            search_space_dsssm["z_dim"],
            search_space_dsssm["h_dim"],
            search_space_dsssm["lamda_b"],
            search_space_dsssm["learning_rate"],
        )
    ]
    print("Number of DSSSM configs:", len(candidate_configs))
    print(candidate_configs)
    return candidate_configs


def run_grid_search(X_train, trainX, trainY, validX, validY, device, n_epochs=40, patience=8, batch_size=64):
    x_dim = X_train.shape[-1]
    y_dim = X_train.shape[-1]
    n_layers = 1

    candidate_configs = build_candidate_configs()
    results = []
    candidate_models = {}
    candidate_histories = {}

    for i, cfg in enumerate(candidate_configs, start=1):
        print("=" * 100)
        print(f"Training DSSSM candidate {i}/{len(candidate_configs)}")
        print(cfg)

        set_seed(3)
        model = DSSSM(
            x_dim=x_dim,
            y_dim=y_dim,
            h_dim=cfg["h_dim"],
            z_dim=cfg["z_dim"],
            d_dim=cfg["d_dim"],
            n_layers=n_layers,
            device=device,
            bidirection=False,
            dataname="SPY",
            lamda_b=cfg["lamda_b"],
            lamda_entropy=0.0,
        ).to(device)
        print("Model device:", next(model.parameters()).device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        history = train_dsssm_model(
            model=model,
            optimizer=optimizer,
            trainX=trainX,
            trainY=trainY,
            validX=validX,
            validY=validY,
            n_epochs=n_epochs,
            batch_size=batch_size,
            patience=patience,
            verbose=True,
        )

        states_val, probs_val = extract_dsssm_regimes(model, validX, validY)
        diag_val = regime_diagnostics(states_val, n_states=cfg["d_dim"])
        daily_summary, _ = daily_segment_summary(states_val)

        usage = diag_val["usage"]
        trans = diag_val["transition_matrix"]
        best_val_loss = float(np.min(history["val_loss"]))
        best_epoch = int(np.argmin(history["val_loss"]) + 1)

        row = {
            "d_dim": cfg["d_dim"],
            "z_dim": cfg["z_dim"],
            "h_dim": cfg["h_dim"],
            "lamda_b": cfg["lamda_b"],
            "learning_rate": cfg["learning_rate"],
            "best_epoch": best_epoch,
            "val_loss": best_val_loss,
            "val_nll": float(history["val_nll"][best_epoch - 1]),
            "val_kld_cat": float(history["val_kld_cat"][best_epoch - 1]),
            "val_balance": float(history["val_balance"][best_epoch - 1]),
            "min_usage": float(diag_val["min_usage"]),
            "max_usage": float(diag_val["max_usage"]),
            "eff_regimes": float(diag_val["effective_regimes"]),
            "switch_rate": float(diag_val["switch_rate"]),
            "avg_transition_diag": float(np.mean(np.diag(trans))),
            **daily_summary,
        }
        for k in range(cfg["d_dim"]):
            row[f"usage_{k}"] = float(usage[k])

        results.append(row)
        key = (
            int(cfg["d_dim"]),
            int(cfg["z_dim"]),
            int(cfg["h_dim"]),
            float(cfg["lamda_b"]),
            float(cfg["learning_rate"]),
        )
        candidate_models[key] = copy.deepcopy(model).cpu()
        candidate_histories[key] = history

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "dsssm_2state_grid_results.csv", index=False)
    print(results_df)
    return results_df, candidate_models, candidate_histories


def select_model(results_df):
    selection_df = results_df[
        (results_df["min_usage"] >= 0.25)
        & (results_df["eff_regimes"] >= 1.70)
        & (results_df["switch_rate"] >= 0.003)
        & (results_df["switch_rate"] <= 0.050)
        & (results_df["median_segments"] >= 1)
        & (results_df["median_segments"] <= 16)
        & (results_df["p90_segments"] <= 28)
    ].copy()

    if len(selection_df) == 0:
        print("No strict admissible DSSSM models. Using relaxed criteria.")
        selection_df = results_df[
            (results_df["min_usage"] >= 0.15)
            & (results_df["eff_regimes"] >= 1.50)
            & (results_df["switch_rate"] >= 0.001)
            & (results_df["switch_rate"] <= 0.070)
            & (results_df["median_segments"] <= 22)
            & (results_df["p90_segments"] <= 34)
        ].copy()

    selection_df = selection_df.sort_values(["val_loss", "switch_rate"], ascending=[True, True])
    selection_df.to_csv(OUT_DIR / "dsssm_2state_selection_results.csv", index=False)
    print(selection_df)

    best_row = selection_df.iloc[0]
    best_key = (
        int(best_row["d_dim"]),
        int(best_row["z_dim"]),
        int(best_row["h_dim"]),
        float(best_row["lamda_b"]),
        float(best_row["learning_rate"]),
    )
    print("Selected DSSSM key:", best_key)
    return best_key, best_row, selection_df


def evaluate_test(best_key, best_row, candidate_models, testX, testY, device):
    final_model = candidate_models[best_key].to(device)
    print("Final model device:", next(final_model.parameters()).device)

    states_test, probs_test = extract_dsssm_regimes(final_model, testX, testY)
    K = int(best_row["d_dim"])
    diag_test = regime_diagnostics(states_test, n_states=K)
    daily_summary, daily_segments = daily_segment_summary(states_test)

    print("DSSSM test usage:", diag_test["usage"])
    print("DSSSM effective regimes:", diag_test["effective_regimes"])
    print("DSSSM switch rate:", diag_test["switch_rate"])
    print("DSSSM transition matrix:")
    print(diag_test["transition_matrix"])
    print("DSSSM daily summary:")
    print(daily_summary)
    print(daily_segments["segments"].value_counts().sort_index())

    np.save(OUT_DIR / "dsssm_2state_test_states.npy", states_test)
    np.save(OUT_DIR / "dsssm_2state_test_probs.npy", probs_test)
    daily_segments.to_csv(OUT_DIR / "dsssm_2state_test_daily_segments.csv", index=False)

    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "best_key": best_key,
            "best_row": best_row.to_dict(),
        },
        OUT_DIR / "dsssm_2state_selected_model.pt",
    )
    return final_model, states_test, probs_test, diag_test, daily_summary, daily_segments


if __name__ == "__main__":
    device = get_device()
    set_seed(3)
    X_train, X_val, X_test = load_arrays()
    trainX, trainY, validX, validY, testX, testY = to_dsssm_tensors(X_train, X_val, X_test, device)
    results_df, candidate_models, candidate_histories = run_grid_search(
        X_train, trainX, trainY, validX, validY, device, n_epochs=40, patience=8, batch_size=64
    )
    best_key, best_row, selection_df = select_model(results_df)
    evaluate_test(best_key, best_row, candidate_models, testX, testY, device)
