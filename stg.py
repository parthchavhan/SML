"""Stochastic gates (Yamada et al., ICML 2020), same folds and feature sets as experiment.py.

This file does not train logistic regression or the random forest. Those stay in the
local experiment.py run. Results go to their own folder, results/stg by default.

One searched setting: the gate penalty lam, grid (0.01, 0.1, 1.0). The middle value
is the one the tuning rule wants chosen most often. Epochs, hidden width, and the
gate noise scale are fixed. There is no early stopping and no validation set inside
a fit.

    python stg.py --check                 # tiny synthetic fit, no parquet
    python stg.py --smoke                 # fold 0, temporal family only
    python stg.py                         # 11 feature sets x 10 folds, resumable

On Kaggle, put this file next to config.py, metrics.py, and experiment.py, with the
GPU left off if it is unavailable. The parquet is found under /kaggle/input.
"""
import math
import sys
import time
import argparse
from pathlib import Path

# A Kaggle dataset is not on the import path until its folder is added.
_kaggle_input = Path("/kaggle/input")
if _kaggle_input.exists():
    for _folder in _kaggle_input.iterdir():
        if _folder.is_dir() and str(_folder) not in sys.path:
            sys.path.insert(0, str(_folder))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

import config as C
import experiment
import metrics

NAME = "stg"
LAM_GRID = (0.01, 0.1, 1.0)
SIGMA = 0.5
HIDDEN = 64
EPOCHS = 30
BATCH = 1024
ADAM_LR = 1e-3


def resolve_data_path(path):
    if path:
        candidate = Path(path)
        if not candidate.exists():
            raise SystemExit(f"features.parquet not found at {candidate}")
        return candidate
    candidates = [Path("features.parquet"), C.DERIVED_DIR / "features.parquet"]
    if _kaggle_input.exists():
        candidates.extend(sorted(_kaggle_input.glob("*/features.parquet")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = " or ".join(str(p) for p in candidates)
    raise SystemExit(f"features.parquet not found at {joined}")


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed():
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(C.SEED)


def gaussian_cdf(x):
    """Standard normal CDF. Used for the probability that a gate is open."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class StochasticGateNet(nn.Module):
    """One gate per input column, then a one-hidden-layer classifier.

    In training the gate is mu + sigma * noise, clipped to [0, 1].
    At test time the noise is off and the gate is clipped mu.
    """

    def __init__(self, n_features, n_classes, sigma=SIGMA, hidden=HIDDEN):
        super().__init__()
        self.sigma = float(sigma)
        self.mu = nn.Parameter(torch.full((n_features,), 0.5))
        self.hidden = nn.Linear(n_features, hidden)
        self.output = nn.Linear(hidden, n_classes)

    def gates(self):
        if self.training:
            noise = torch.randn_like(self.mu)
            raw = self.mu + self.sigma * noise
        else:
            raw = self.mu
        return raw.clamp(0.0, 1.0)

    def penalty(self):
        # The paper sums the CDFs. The mean keeps one lam grid comparable
        # across a 12-column family and the full 216-column set.
        return gaussian_cdf(self.mu / self.sigma).mean()

    def forward(self, x):
        gated = x * self.gates()
        return self.output(torch.relu(self.hidden(gated)))


def _class_weights(y_local, n_classes):
    counts = np.bincount(y_local, minlength=n_classes).astype(np.float64)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return weights.astype(np.float32)


def fit_stg(X_train, y_train, others, n_species, lam, class_weight=True, epochs=EPOCHS):
    """Fit on the training rows only. `others` are transformed with that same fit."""
    if np.any(np.asarray(y_train) < 0):
        raise SystemExit("training labels include a non-focal row")
    _seed()
    dev = device()
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=int)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    transformed = [scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32) for X in others]

    present = np.unique(y_train)
    local = {int(cls): i for i, cls in enumerate(present)}
    y_local = np.array([local[int(v)] for v in y_train], dtype=np.int64)
    n_local = len(present)

    model = StochasticGateNet(X_train.shape[1], n_local).to(dev)
    weight = None
    if class_weight:
        weight = torch.tensor(_class_weights(y_local, n_local), device=dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    xt = torch.tensor(X_train, device=dev)
    yt = torch.tensor(y_local, device=dev)
    n_train = len(y_local)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, BATCH):
            batch = order[start:start + BATCH]
            optimiser.zero_grad()
            logits = model(xt[batch])
            task = nn.functional.cross_entropy(logits, yt[batch], weight=weight)
            # Scale by the batch share so one epoch applies lam once, not once per batch.
            loss = task + float(lam) * model.penalty() * (len(batch) / n_train)
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        open_gates = float((model.mu.detach().clamp(0.0, 1.0) > 0.5).float().mean())
        scores = []
        for block in transformed:
            logits = model(torch.tensor(block, device=dev))
            local_proba = torch.softmax(logits, dim=1).cpu().numpy()
            full = np.zeros((len(block), n_species), dtype=np.float64)
            full[:, present] = local_proba
            scores.append(full)
    return scores, open_gates


def select_lambda(data, cols, outer_k, epochs):
    """Inner CV on focal clips whose fold is not outer_k. Returns value and three scores."""
    groups = experiment.inner_groups(outer_k)
    focal = data["domain"] == "focal"
    X = data["X"]
    means = []
    for value in LAM_GRID:
        fold_scores = []
        for val_folds in groups:
            train_folds = [fold for group in groups if group is not val_folds for fold in group]
            if outer_k in train_folds or outer_k in val_folds:
                raise SystemExit("outer test fold leaked into inner CV")
            tr = experiment._rows(focal & np.isin(data["focal_fold"], train_folds))
            va = experiment._rows(focal & np.isin(data["focal_fold"], val_folds))
            if np.any(data["domain"][tr] != "focal") or np.any(data["domain"][va] != "focal"):
                raise SystemExit("soundscape clip entered inner CV")
            (Sv,), _ = fit_stg(
                X[tr][:, cols], data["y"][tr], [X[va][:, cols]],
                data["n_species"], value, epochs=epochs,
            )
            fold_scores.append(experiment._macro_on_present(data["Y"][va], Sv))
        means.append(float(np.mean(fold_scores)))
        shown = ", ".join(f"{s:.4f}" for s in fold_scores)
        print(f"  inner lam={value} -> {means[-1]:.4f} ({shown})", flush=True)
    return experiment.choose_param(LAM_GRID, means), means


def evaluate_fold(data, cols, outer_k, lam, epochs):
    """Refit on the nine focal folds and score both test domains."""
    focal = data["domain"] == "focal"
    tr = experiment._rows(focal & (data["focal_fold"] != outer_k))
    foc = experiment._rows(focal & (data["focal_fold"] == outer_k))
    sca = experiment._rows((data["domain"] == "soundscape") & (data["scape_fold"] == outer_k))
    if np.any(data["domain"][tr] != "focal"):
        raise SystemExit("soundscape clip entered training")
    X = data["X"]
    (Sf, Ss), open_gates = fit_stg(
        X[tr][:, cols], data["y"][tr], [X[foc][:, cols], X[sca][:, cols]],
        data["n_species"], lam, epochs=epochs,
    )
    Yf, Ys = data["Y"][foc], data["Y"][sca]
    bird = Ys.any(axis=1)
    return {
        "species": data["species"],
        "focal_pos": Yf.sum(axis=0), "scape_pos": Ys.sum(axis=0),
        "graded": metrics.graded_species(Yf.sum(axis=0), Ys.sum(axis=0)),
        "focal": experiment.score_split(Yf, Sf),
        "scape": experiment.score_split(Ys, Ss),
        "bird": experiment.score_split(Ys[bird], Ss[bird]),
        "focal_top1": experiment.top1(Yf, Sf, np.ones(len(foc), dtype=bool)),
        "scape_top1": experiment.top1(Ys, Ss, bird),
        "confused_focal": experiment.confusion_counts(Yf, Sf, data["species"], np.ones(len(foc), dtype=bool)),
        "confused_scape": experiment.confusion_counts(Ys, Ss, data["species"], Ys.sum(axis=1) == 1),
        "gates_open": open_gates,
    }


def run_fold(data, feature_set, outer_k, results, epochs):
    cols = data["sets"][feature_set]
    runs_path, species_path, confused_path = experiment.part_paths(results, NAME, outer_k)
    print(f"[{NAME} fold {outer_k}] {feature_set} {len(cols)} columns", flush=True)
    started = time.perf_counter()
    lam, inner_scores = select_lambda(data, cols, outer_k, epochs)
    print(f"  chosen lam={lam}", flush=True)
    scored = evaluate_fold(data, cols, outer_k, lam, epochs)
    graded = scored["graded"]
    species_rows = []
    for i, sp in enumerate(scored["species"]):
        species_rows.append({
            "algorithm": NAME, "feature_set": feature_set, "fold": outer_k, "species": sp,
            "n_focal_pos": int(scored["focal_pos"][i]),
            "n_scape_pos": int(scored["scape_pos"][i]),
            "focal_auroc": scored["focal"]["auroc"][i], "focal_ap": scored["focal"]["ap"][i],
            "scape_auroc": scored["scape"]["auroc"][i], "scape_ap": scored["scape"]["ap"][i],
            "scape_bird_auroc": scored["bird"]["auroc"][i], "scape_bird_ap": scored["bird"]["ap"][i],
        })
    confused_rows = []
    for domain, pairs in (("focal", scored["confused_focal"]), ("soundscape", scored["confused_scape"])):
        if feature_set != "all":
            break
        for pair in pairs:
            confused_rows.append({
                "algorithm": NAME, "feature_set": feature_set, "fold": outer_k,
                "domain": domain, **pair,
            })
    experiment._append(species_path, species_rows)
    experiment._append(confused_path, confused_rows)
    experiment._append(runs_path, [{
        "algorithm": NAME, "feature_set": feature_set, "fold": outer_k,
        "n_columns": int(len(cols)), "param": "lam", "param_value": lam,
        "inner_0": inner_scores[0], "inner_1": inner_scores[1], "inner_2": inner_scores[2],
        "n_graded": int(len(graded)),
        "gates_open": scored["gates_open"],
        "focal_auroc": metrics.macro(scored["focal"]["auroc"], graded),
        "scape_auroc": metrics.macro(scored["scape"]["auroc"], graded),
        "scape_bird_auroc": metrics.macro(scored["bird"]["auroc"], graded),
        "focal_ap": metrics.macro(scored["focal"]["ap"], graded),
        "scape_ap": metrics.macro(scored["scape"]["ap"], graded),
        "scape_bird_ap": metrics.macro(scored["bird"]["ap"], graded),
        "focal_top1": scored["focal_top1"], "scape_top1": scored["scape_top1"],
        "seconds": time.perf_counter() - started,
    }])
    print(f"  focal {metrics.macro(scored['focal']['auroc'], graded):.4f}  "
          f"soundscape {metrics.macro(scored['scape']['auroc'], graded):.4f}  "
          f"gates open {scored['gates_open']:.2f}", flush=True)


def check_model():
    """A few synthetic rows. Confirms the probability layout, not a real score."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(C.SEED)
    X = rng.normal(size=(24, 5)).astype(np.float64)
    y = np.array([0, 1, 2, 0, 1, 2] * 4)
    (S,), open_gates = fit_stg(X, y, [X], 4, 0.1, class_weight=True, epochs=2)
    if S.shape != (24, 4):
        raise SystemExit(f"scores have shape {S.shape}, expected (24, 4)")
    if not np.allclose(S.sum(axis=1), 1.0, atol=1e-5):
        raise SystemExit("probabilities do not sum to 1")
    if not np.all(S[:, 3] == 0):
        raise SystemExit("a class absent from training did not stay at score 0")
    if not 0.0 <= open_gates <= 1.0:
        raise SystemExit(f"open-gate fraction is {open_gates}")
    print("stochastic-gates check passed")


def main():
    parser = argparse.ArgumentParser(description="Nested CV for stochastic gates")
    parser.add_argument("--data", default=None)
    parser.add_argument("--results", default="results/stg")
    parser.add_argument("--check", action="store_true", help="Synthetic fit only. Does not read the parquet.")
    parser.add_argument("--smoke", action="store_true", help="Outer fold 0, temporal family only.")
    parser.add_argument("--feature-set", action="append", dest="feature_sets", default=None)
    parser.add_argument("--fold", action="append", dest="folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.check:
        check_model()
        return

    metrics.self_check()
    experiment.check_splits()
    path = resolve_data_path(args.data)
    print(f"stochastic gates on {device()}, lam grid {LAM_GRID}, epochs {args.epochs}", flush=True)
    print(f"loading {path}", flush=True)
    data = experiment.load_experiment(path)
    print(f"{experiment.N_ROWS} rows, {data['n_species']} species, {experiment.N_FEATURES} features", flush=True)

    feature_sets = args.feature_sets or (["temporal"] if args.smoke else experiment.feature_set_order())
    folds = args.folds or ([0] if args.smoke else list(range(C.N_FOLDS)))
    unknown = [name for name in feature_sets if name not in data["sets"]]
    if unknown:
        raise SystemExit(f"unknown feature set: {unknown}")
    for outer_k in folds:
        runs_path, _, _ = experiment.part_paths(args.results, NAME, outer_k)
        done = experiment.done_keys(runs_path)
        for feature_set in feature_sets:
            key = (NAME, feature_set, outer_k)
            if key in done:
                print(f"skip {key}", flush=True)
                continue
            run_fold(data, feature_set, outer_k, args.results, args.epochs)


if __name__ == "__main__":
    main()