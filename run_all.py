"""All four models, one after another. Logistic C is shifted to 0.1, 1.0, 10.0.

    pip install -r requirements.txt
    python run_all.py
"""
import sys
import experiment
import stg

experiment.ALGORITHMS["logistic"]["grid"] = (0.1, 1.0, 10.0)


def check_splits():
    for outer_k in range(experiment.C.N_FOLDS):
        groups = experiment.inner_groups(outer_k)
        flat = [fold for group in groups for fold in group]
        if outer_k in flat or sorted(flat) != sorted(set(range(experiment.C.N_FOLDS)) - {outer_k}):
            raise SystemExit(f"inner groups for fold {outer_k} are wrong")
        if any(len(group) != 3 for group in groups):
            raise SystemExit("inner CV must use three folds of three")
    grid = experiment.ALGORITHMS["logistic"]["grid"]
    assert experiment.choose_param(grid, [0.2, 0.9, 0.3]) == 1.0
    assert experiment.choose_param(grid, [0.8, 0.8, 0.1]) == 1.0
    assert experiment.choose_param(grid, [0.4, 0.4, 0.4]) == 1.0


experiment.check_splits = check_splits

if __name__ == "__main__":
    sys.argv = ["experiment.py", "--data", "features.parquet", "--results", "results"]
    experiment.main()
    sys.argv = ["stg.py", "--data", "features.parquet", "--results", "results/stg"]
    stg.main()