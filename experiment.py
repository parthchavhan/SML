"""Nested cross-validation for the 11 feature sets and 3 algorithms.

Train and tune on focal clips only. Soundscape clips are scored after the
refit and never enter scaling, the inner loop, or model selection.

Algorithms, simple to complex:
  logistic   multinomial logistic regression, tune C
  forest     random forest, tune max_depth
  lightgbm   LightGBM (Ke et al., NIPS 2017), tune learning_rate
             Fixed tree count, no early stopping, no internal validation.

    python experiment.py --pilot          # fold-0 logistic check, not logged
    python experiment.py --smoke          # fold 0, temporal family, all 3 models
    python experiment.py                  # full 11 x 3 x 10 run (resumable)
    python experiment.py --merge          # combine results/parts into the logs

Each (algorithm, fold) writes its own files under results/parts, so folds can
run in separate processes. A finished (algorithm, feature set, fold) row is
skipped on the next launch.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config as C
import metrics

# name, column prefix, width. Prefixes are the family tags in features.parquet.
FAMILIES = (
    ("cepstral", "cep_", 78),
    ("shape", "shp_", 19),
    ("subband", "sub_", 92),
    ("temporal", "tmp_", 12),
    ("pitch", "pit_", 15),
)
FAMILY_NAMES = [name for name, _, _ in FAMILIES]
N_FEATURES = 216
N_ROWS = 39711

# One hyperparameter each, three values, middle value is the one we hope wins.
ALGORITHMS = {
    "logistic": {"param": "C", "grid": (0.01, 0.1, 1.0), "scale": True},
    "forest": {"param": "max_depth", "grid": (8, 16, 32), "scale": False},
    "lightgbm": {"param": "learning_rate", "grid": (0.02, 0.05, 0.1), "scale": False},
}
N_TREES = 200


def feature_set_order():
    names = ["all"] + list(FAMILY_NAMES)
    names += [f"without_{name}" for name in FAMILY_NAMES]
    return names


def resolve_data_path(path):
    if path:
        candidate = Path(path)
        if not candidate.exists():
            raise SystemExit(f"features.parquet not found at {candidate}")
        return candidate
    candidates = [Path("features.parquet"), C.DERIVED_DIR / "features.parquet"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = " or ".join(str(p) for p in candidates)
    raise SystemExit(f"features.parquet not found at {joined}")


def _as_labels(value):
    if value is None:
        return []
    return [str(x) for x in list(value)]


def load_experiment(path):
    """One matrix of features plus the fixed species label matrix."""
    df = pd.read_parquet(path)
    if len(df) != N_ROWS:
        raise SystemExit(f"{path} has {len(df)} rows; expected {N_ROWS}")
    labels = [_as_labels(v) for v in df["labels"].tolist()]
    focal = df["domain"].to_numpy() == "focal"
    if any(len(labs) != 1 for labs, is_focal in zip(labels, focal) if is_focal):
        raise SystemExit("a focal clip does not have exactly one label")
    species = sorted({labs[0] for labs, is_focal in zip(labels, focal) if is_focal})
    index = {sp: i for i, sp in enumerate(species)}
    stray = sorted({lab for labs, is_focal in zip(labels, focal) if not is_focal for lab in labs if lab not in index})
    if stray:
        raise SystemExit(f"soundscape labels outside the focal species list: {stray[:5]}")

    columns = []
    sets = {}
    for name, prefix, width in FAMILIES:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if len(cols) != width:
            raise SystemExit(f"{name} has {len(cols)} columns; expected {width}")
        start = len(columns)
        columns.extend(cols)
        sets[name] = np.arange(start, start + width)
    if len(columns) != N_FEATURES:
        raise SystemExit(f"{len(columns)} feature columns; expected {N_FEATURES}")
    sets["all"] = np.arange(N_FEATURES)
    for name, _, width in FAMILIES:
        keep = np.ones(N_FEATURES, dtype=bool)
        keep[sets[name]] = False
        without = np.flatnonzero(keep)
        if len(without) != N_FEATURES - width:
            raise SystemExit(f"without_{name} has the wrong width")
        sets[f"without_{name}"] = without

    X = df[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise SystemExit("feature matrix contains NaN or inf")
    Y = np.zeros((len(df), len(species)), dtype=bool)
    y = np.full(len(df), -1, dtype=int)
    for i, labs in enumerate(labels):
        for lab in labs:
            Y[i, index[lab]] = True
        if focal[i]:
            y[i] = index[labs[0]]
    focal_fold = df["focal_fold"].to_numpy(dtype=int)
    scape_fold = df["scape_fold"].to_numpy(dtype=int)
    if not set(focal_fold[focal]).issubset(set(range(C.N_FOLDS))):
        raise SystemExit("focal clips are not assigned to folds 0-9")
    sound = ~focal
    if not set(scape_fold[sound]).issubset(set(range(C.N_FOLDS))):
        raise SystemExit("soundscape clips are not assigned to folds 0-9")
    return {
        "X": X, "Y": Y, "y": y, "domain": df["domain"].to_numpy(),
        "focal_fold": focal_fold, "scape_fold": scape_fold,
        "species": species, "sets": sets, "n_species": len(species),
    }


def inner_groups(outer_k, n_folds=C.N_FOLDS):
    """Nine training folds, sorted, then three chunks of three."""
    folds = [i for i in range(n_folds) if i != outer_k]
    return [folds[0:3], folds[3:6], folds[6:9]]


def make_estimator(name, param, class_weight=True, n_estimators=N_TREES, n_jobs=-1):
    """Library models only. Nothing here searches, stops early, or scores itself."""
    if name == "logistic":
        return LogisticRegression(
            C=float(param), solver="lbfgs", max_iter=1000,
            class_weight="balanced" if class_weight else None,
            random_state=C.SEED,
        )
    if name == "forest":
        return RandomForestClassifier(
            n_estimators=int(n_estimators), max_depth=int(param), min_samples_leaf=2,
            class_weight="balanced_subsample" if class_weight else None,
            random_state=C.SEED, n_jobs=n_jobs,
        )
    if name == "lightgbm":
        # n_estimators is fixed. No early_stopping callback and no valid sets.
        return LGBMClassifier(
            n_estimators=int(n_estimators), learning_rate=float(param), num_leaves=31,
            objective="multiclass",
            class_weight="balanced" if class_weight else None,
            random_state=C.SEED, n_jobs=n_jobs, verbosity=-1,
        )
    raise KeyError(name)


def align_proba(estimator, X, n_species):
    """Probability columns in species-index order. A class absent from training gets 0."""
    proba = estimator.predict_proba(X)
    out = np.zeros((len(X), n_species), dtype=np.float64)
    for j, cls in enumerate(estimator.classes_):
        out[:, int(cls)] = proba[:, j]
    return out


def _as_frame(X):
    """Named columns, shared by fit and predict, so LightGBM does not warn on every call."""
    X = np.asarray(X, dtype=np.float64)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


def fit_and_score(name, param, X_train, y_train, others, n_species, class_weight=True,
                  n_estimators=N_TREES, n_jobs=-1):
    """Fit on the training rows only. `others` are transformed with that same fit."""
    if np.any(np.asarray(y_train) < 0):
        raise SystemExit("training labels include a non-focal row")
    X_train = np.asarray(X_train, dtype=np.float64)
    frames = []
    if ALGORITHMS[name]["scale"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        frames = [scaler.transform(np.asarray(X, dtype=np.float64)) for X in others]
    else:
        frames = [np.asarray(X, dtype=np.float64) for X in others]
    estimator = make_estimator(name, param, class_weight, n_estimators, n_jobs)
    estimator.fit(_as_frame(X_train), y_train)
    return [align_proba(estimator, _as_frame(X), n_species) for X in frames]


def choose_param(grid, means):
    """Highest inner mean. An exact tie resolves to the middle grid value."""
    means = np.asarray(means, dtype=float)
    best = np.nanmax(means)
    winners = [i for i, value in enumerate(means) if np.isclose(value, best)]
    middle = (len(grid) - 1) / 2
    chosen = min(winners, key=lambda i: (abs(i - middle), i))
    return grid[chosen]


def _rows(mask):
    return np.flatnonzero(mask)


def _macro_on_present(Y, S):
    present = np.flatnonzero(Y.sum(axis=0) > 0)
    return metrics.macro(metrics.per_species(metrics.auroc, Y, S), present)


def select_hyperparameter(name, data, cols, outer_k, n_jobs=-1):
    """Inner CV on focal clips whose fold is not outer_k. Returns value and three scores."""
    spec = ALGORITHMS[name]
    groups = inner_groups(outer_k)
    focal = data["domain"] == "focal"
    means = []
    X = data["X"]
    for value in spec["grid"]:
        fold_scores = []
        for val_folds in groups:
            train_folds = [fold for group in groups if group is not val_folds for fold in group]
            if outer_k in train_folds or outer_k in val_folds:
                raise SystemExit("outer test fold leaked into inner CV")
            tr = _rows(focal & np.isin(data["focal_fold"], train_folds))
            va = _rows(focal & np.isin(data["focal_fold"], val_folds))
            if np.any(data["domain"][tr] != "focal") or np.any(data["domain"][va] != "focal"):
                raise SystemExit("soundscape clip entered inner CV")
            (Sv,) = fit_and_score(
                name, value, X[tr][:, cols], data["y"][tr], [X[va][:, cols]],
                data["n_species"], n_jobs=n_jobs,
            )
            fold_scores.append(_macro_on_present(data["Y"][va], Sv))
        means.append(float(np.mean(fold_scores)))
        shown = ", ".join(f"{s:.4f}" for s in fold_scores)
        print(f"  inner {spec['param']}={value} -> {means[-1]:.4f} ({shown})", flush=True)
    return choose_param(spec["grid"], means), means


def top1(Y, S, mask):
    """Share of clips whose highest-scoring species is one of the true labels."""
    rows = _rows(mask)
    if len(rows) == 0:
        return np.nan
    pred = S[rows].argmax(axis=1)
    hit = Y[rows, pred]
    return float(hit.mean())


def confusion_counts(Y, S, species, mask):
    """Non-zero (true, predicted) counts on single-label clips where the two differ."""
    rows = _rows(mask)
    if len(rows) == 0:
        return []
    pred = S[rows].argmax(axis=1)
    true = Y[rows].argmax(axis=1)
    single = Y[rows].sum(axis=1) == 1
    wrong = single & (true != pred)
    if not np.any(wrong):
        return []
    pairs, counts = np.unique(np.stack([true[wrong], pred[wrong]]), axis=1, return_counts=True)
    return [
        {"true_species": species[int(a)], "pred_species": species[int(b)], "count": int(n)}
        for (a, b), n in zip(pairs.T, counts)
    ]


def score_split(Y, S):
    return {
        "auroc": metrics.per_species(metrics.auroc, Y, S),
        "ap": metrics.per_species(metrics.average_precision, Y, S),
    }


def evaluate_fold(data, cols, outer_k, name, param, class_weight=True, n_jobs=-1):
    """Refit on the nine focal folds and score both test domains."""
    focal = data["domain"] == "focal"
    tr = _rows(focal & (data["focal_fold"] != outer_k))
    foc = _rows(focal & (data["focal_fold"] == outer_k))
    sca = _rows((data["domain"] == "soundscape") & (data["scape_fold"] == outer_k))
    if np.any(data["domain"][tr] != "focal"):
        raise SystemExit("soundscape clip entered training")
    X = data["X"]
    Sf, Ss = fit_and_score(
        name, param, X[tr][:, cols], data["y"][tr], [X[foc][:, cols], X[sca][:, cols]],
        data["n_species"], class_weight=class_weight, n_jobs=n_jobs,
    )
    Yf, Ys = data["Y"][foc], data["Y"][sca]
    focal_scores = score_split(Yf, Sf)
    scape_scores = score_split(Ys, Ss)
    bird = Ys.any(axis=1)
    bird_scores = score_split(Ys[bird], Ss[bird])
    focal_pos = Yf.sum(axis=0)
    scape_pos = Ys.sum(axis=0)
    graded = metrics.graded_species(focal_pos, scape_pos)
    return {
        "species": data["species"],
        "focal_pos": focal_pos, "scape_pos": scape_pos, "graded": graded,
        "focal": focal_scores, "scape": scape_scores, "bird": bird_scores,
        "focal_top1": top1(Yf, Sf, np.ones(len(foc), dtype=bool)),
        "scape_top1": top1(Ys, Ss, bird),
        "confused_focal": confusion_counts(Yf, Sf, data["species"], np.ones(len(foc), dtype=bool)),
        "confused_scape": confusion_counts(Ys, Ss, data["species"], Ys.sum(axis=1) == 1),
    }


def _append(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def done_keys(path):
    path = Path(path)
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return set(zip(df.algorithm, df.feature_set, df.fold.astype(int)))


def part_paths(results, name, outer_k):
    directory = Path(results) / "parts"
    stem = f"{name}_fold{outer_k}"
    return (
        directory / f"{stem}_runs.csv",
        directory / f"{stem}_species.csv",
        directory / f"{stem}_confused.csv",
    )


def run_fold(data, name, feature_set, outer_k, results, n_jobs=-1):
    cols = data["sets"][feature_set]
    runs_path, species_path, confused_path = part_paths(results, name, outer_k)
    print(f"[{name} fold {outer_k}] {feature_set} {len(cols)} columns", flush=True)
    started = time.perf_counter()
    param, inner_scores = select_hyperparameter(name, data, cols, outer_k, n_jobs=n_jobs)
    print(f"  chosen {ALGORITHMS[name]['param']}={param}", flush=True)
    scored = evaluate_fold(data, cols, outer_k, name, param, n_jobs=n_jobs)
    graded = scored["graded"]
    species_rows = []
    for i, sp in enumerate(scored["species"]):
        species_rows.append({
            "algorithm": name, "feature_set": feature_set, "fold": outer_k, "species": sp,
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
                "algorithm": name, "feature_set": feature_set, "fold": outer_k,
                "domain": domain, **pair,
            })
    _append(species_path, species_rows)
    _append(confused_path, confused_rows)
    _append(runs_path, [{
        "algorithm": name, "feature_set": feature_set, "fold": outer_k,
        "n_columns": int(len(cols)), "param": ALGORITHMS[name]["param"],
        "param_value": param,
        "inner_0": inner_scores[0], "inner_1": inner_scores[1], "inner_2": inner_scores[2],
        "n_graded": int(len(graded)),
        "focal_auroc": metrics.macro(scored["focal"]["auroc"], graded),
        "scape_auroc": metrics.macro(scored["scape"]["auroc"], graded),
        "scape_bird_auroc": metrics.macro(scored["bird"]["auroc"], graded),
        "focal_ap": metrics.macro(scored["focal"]["ap"], graded),
        "scape_ap": metrics.macro(scored["scape"]["ap"], graded),
        "scape_bird_ap": metrics.macro(scored["bird"]["ap"], graded),
        "focal_top1": scored["focal_top1"], "scape_top1": scored["scape_top1"],
        "seconds": time.perf_counter() - started,
    }])


def run_pilot(data):
    """Jaime's pipeline check: logistic C=0.1, no class weight, outer fold 0, all 216 columns.

    Printed only. Not written to the results log, and not a number for the report.
    """
    print("pilot: logistic C=0.1, no class weight, all features, outer fold 0", flush=True)
    started = time.perf_counter()
    focal = data["domain"] == "focal"
    tr = _rows(focal & (data["focal_fold"] != 0))
    foc = _rows(focal & (data["focal_fold"] == 0))
    sca = _rows((data["domain"] == "soundscape") & (data["scape_fold"] == 0))
    cols = data["sets"]["all"]
    X = data["X"]
    Sf, Ss = fit_and_score(
        "logistic", 0.1, X[tr][:, cols], data["y"][tr], [X[foc][:, cols], X[sca][:, cols]],
        data["n_species"], class_weight=False,
    )
    Yf, Ys = data["Y"][foc], data["Y"][sca]
    graded = metrics.graded_species(Yf.sum(axis=0), Ys.sum(axis=0))
    focal_auroc = metrics.macro(metrics.per_species(metrics.auroc, Yf, Sf), graded)
    scape_auroc = metrics.macro(metrics.per_species(metrics.auroc, Ys, Ss), graded)
    print(f"pilot graded species: {len(graded)} (Jaime's check reported 39)")
    print(f"pilot focal macro AUROC:     {focal_auroc:.4f} (about 0.94)")
    print(f"pilot soundscape macro AUROC: {scape_auroc:.4f} (about 0.80)")
    print(f"pilot seconds: {time.perf_counter() - started:.0f}")


def check_score_alignment():
    """Every model must return one column per species, with a zero for a class it never saw."""
    rng = np.random.default_rng(C.SEED)
    X = rng.normal(size=(6, 4))
    y = np.array([0, 1, 2, 0, 1, 2])  # class 3 and 4 never seen; 5 species requested
    for name, param in (("logistic", 1.0), ("forest", 2), ("lightgbm", 0.1)):
        n_estimators = 8 if name != "logistic" else N_TREES
        (S,) = fit_and_score(
            name, param, X, y, [X], 5, class_weight=False, n_estimators=n_estimators, n_jobs=1,
        )
        if S.shape != (6, 5):
            raise SystemExit(f"{name} scores have shape {S.shape}, expected (6, 5)")
        if not np.all(S[:, 3] == 0) or not np.all(S[:, 4] == 0):
            raise SystemExit(f"{name} did not leave the unseen class at score 0")
        if not np.allclose(S.sum(axis=1), 1):
            raise SystemExit(f"{name} probabilities do not sum to 1")
    print("score-alignment check passed")


def check_splits():
    for outer_k in range(C.N_FOLDS):
        groups = inner_groups(outer_k)
        flat = [fold for group in groups for fold in group]
        if outer_k in flat or sorted(flat) != sorted(set(range(C.N_FOLDS)) - {outer_k}):
            raise SystemExit(f"inner groups for fold {outer_k} are not a partition of the training folds")
        if any(len(group) != 3 for group in groups):
            raise SystemExit("inner CV must use three folds of three")
    grid = ALGORITHMS["logistic"]["grid"]
    assert choose_param(grid, [0.2, 0.9, 0.3]) == 0.1
    assert choose_param(grid, [0.8, 0.8, 0.1]) == 0.1  # tie resolves toward the middle
    assert choose_param(grid, [0.4, 0.4, 0.4]) == 0.1


def merge_parts(results):
    """Concatenate per-fold logs. Later rows win if a fold was resumed."""
    results = Path(results)
    parts = results / "parts"
    mapping = {
        "runs.csv": "*_runs.csv",
        "species_scores.csv": "*_species.csv",
        "confused_counts.csv": "*_confused.csv",
    }
    keys = {
        "runs.csv": ["algorithm", "feature_set", "fold"],
        "species_scores.csv": ["algorithm", "feature_set", "fold", "species"],
        "confused_counts.csv": ["algorithm", "feature_set", "fold", "domain", "true_species", "pred_species"],
    }
    if not parts.exists():
        raise SystemExit(f"no part logs in {parts}")
    for dest, pattern in mapping.items():
        files = sorted(parts.glob(pattern))
        frames = [pd.read_csv(path) for path in files if path.stat().st_size > 0]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(keys[dest], keep="last")
        df.to_csv(results / dest, index=False)
        print(f"merged {len(df)} rows -> {results / dest}")


def main():
    parser = argparse.ArgumentParser(description="Nested CV on features.parquet")
    parser.add_argument("--data", default=None)
    parser.add_argument("--results", default="results")
    parser.add_argument("--pilot", action="store_true",
                        help="Fold-0 logistic check (C=0.1, no class weight). Not logged.")
    parser.add_argument("--smoke", action="store_true",
                        help="Outer fold 0, temporal family, all three algorithms.")
    parser.add_argument("--merge", action="store_true", help="Combine results/parts into the summary logs.")
    parser.add_argument("--algo", action="append", default=None)
    parser.add_argument("--feature-set", action="append", dest="feature_sets", default=None)
    parser.add_argument("--fold", action="append", dest="folds", type=int, default=None)
    parser.add_argument("--tree-jobs", type=int, default=-1,
                        help="Cores per tree model. Use 1 when several folds run at once.")
    args = parser.parse_args()
    if args.merge:
        merge_parts(args.results)
        return
    metrics.self_check()
    check_splits()
    check_score_alignment()
    path = resolve_data_path(args.data)
    print(f"loading {path}", flush=True)
    data = load_experiment(path)
    print(f"{N_ROWS} rows, {data['n_species']} species, {N_FEATURES} features", flush=True)
    if args.pilot:
        run_pilot(data)
        return

    algos = args.algo or list(ALGORITHMS)
    feature_sets = args.feature_sets or (["temporal"] if args.smoke else feature_set_order())
    folds = args.folds or ([0] if args.smoke else list(range(C.N_FOLDS)))
    unknown_algo = [name for name in algos if name not in ALGORITHMS]
    unknown_set = [name for name in feature_sets if name not in data["sets"]]
    if unknown_algo:
        raise SystemExit(f"unknown algorithm: {unknown_algo}")
    if unknown_set:
        raise SystemExit(f"unknown feature set: {unknown_set}")
    for name in algos:
        for outer_k in folds:
            runs_path, _, _ = part_paths(args.results, name, outer_k)
            done = done_keys(runs_path)
            for feature_set in feature_sets:
                key = (name, feature_set, outer_k)
                if key in done:
                    print(f"skip {key}", flush=True)
                    continue
                run_fold(data, name, feature_set, outer_k, args.results, n_jobs=args.tree_jobs)
    # A filtered launch is one piece of a parallel run. Merge only a complete single process.
    if args.algo is None and args.folds is None and not args.smoke:
        merge_parts(args.results)


if __name__ == "__main__":
    main()
