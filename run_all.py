"""LightGBM on CUDA, CatBoost on GPU, then stochastic gates on the shifted lam grid.

    pip install catboost
    python run_all.py
"""
import sys
from pathlib import Path

import experiment
import stg


def make_estimator(name, param, class_weight=True, n_estimators=experiment.N_TREES, n_jobs=-1):
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=int(n_estimators), learning_rate=float(param), num_leaves=31,
            objective="multiclass", device_type="cuda",
            class_weight="balanced" if class_weight else None,
            random_state=experiment.C.SEED, n_jobs=1, verbosity=-1,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=int(n_estimators), learning_rate=float(param), depth=6,
            loss_function="MultiClass", boosting_type="Plain",
            auto_class_weights="Balanced" if class_weight else None,
            task_type="GPU", devices="0",
            random_seed=experiment.C.SEED, verbose=False, allow_writing_files=False,
        )
    return _orig(name, param, class_weight, n_estimators, n_jobs)


_orig = experiment.make_estimator
experiment.make_estimator = make_estimator
experiment.ALGORITHMS["catboost"] = {
    "param": "learning_rate", "grid": (0.03, 0.1, 0.3), "scale": False,
}
stg.LAM_GRID = (0.001, 0.01, 0.1)


if __name__ == "__main__":
    data = "features.parquet"
    if not Path(data).exists():
        raise SystemExit("put features.parquet in this folder first")
    sys.argv = ["experiment.py", "--algo", "lightgbm", "--algo", "catboost",
                "--data", data, "--results", "results/gpu"]
    experiment.main()
    experiment.merge_parts("results/gpu")
    sys.argv = ["stg.py", "--data", data, "--results", "results/stg_shift"]
    stg.main()