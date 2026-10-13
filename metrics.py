"""Ranking metrics written from scratch.

The AUROC and average-precision formulas are the ones in docs/TECHNICAL.md.
Nothing in this module calls sklearn.metrics.
"""
import numpy as np


def auroc(y, s):
    """Mann-Whitney AUROC. Ties get average ranks. Undefined if either class is empty."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return np.nan
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    ranks = np.empty(len(s))
    ranks[np.argsort(s, kind="mergesort")] = np.arange(1, len(s) + 1)
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def average_precision(y, s):
    """Average precision. Tied scores keep their original order (stable sort)."""
    y = np.asarray(y, dtype=bool)
    if int(y.sum()) == 0 or len(y) == 0:
        return np.nan
    y = y[np.argsort(-np.asarray(s, dtype=float), kind="mergesort")]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float(precision[y].mean())


def per_species(metric, Y, S):
    """One metric per species column. Y and S are (clips, species)."""
    Y = np.asarray(Y)
    S = np.asarray(S, dtype=float)
    return np.array([metric(Y[:, c], S[:, c]) for c in range(Y.shape[1])], dtype=float)


def macro(values, species):
    """Unweighted mean over a species index set. Empty or all-missing is NaN."""
    values = np.asarray(values, dtype=float)
    idx = np.asarray(list(species), dtype=int)
    if idx.size == 0:
        return np.nan
    picked = values[idx]
    if np.all(np.isnan(picked)):
        return np.nan
    return float(np.nanmean(picked))


def graded_species(focal_pos, scape_pos, min_scape=5, min_focal=1):
    """Species graded in one outer fold: enough positives in both test domains."""
    focal_pos = np.asarray(focal_pos)
    scape_pos = np.asarray(scape_pos)
    return np.flatnonzero((scape_pos >= min_scape) & (focal_pos >= min_focal))


def sample_sd(values):
    """Sample standard deviation across outer folds (ddof=1)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.nan
    return float(np.std(values, ddof=1))


def robustness(focal_mean, sound_mean):
    """Share of above-chance skill kept. Undefined if focal skill is exactly zero."""
    focal_mean = float(focal_mean)
    sound_mean = float(sound_mean)
    if focal_mean == 0.5:
        return np.nan
    return (sound_mean - 0.5) / (focal_mean - 0.5)


def family_summary(focal_folds, sound_folds):
    """Informativeness, robustness, and the no-skill rule from one feature set.

    I and R use the fold means. R's error bar is the SD of the per-fold ratios.
    SD is the sample standard deviation (ddof=1). A family whose focal mean minus
    one SD is at or below 0.5 has no skill to keep.
    """
    focal = np.asarray(focal_folds, dtype=float)
    sound = np.asarray(sound_folds, dtype=float)
    I = float(np.mean(focal))
    I_sd = sample_sd(focal)
    sound_mean = float(np.mean(sound))
    sound_sd = sample_sd(sound)
    raw_drop = I - sound_mean
    if (I - I_sd) <= 0.5:
        return {
            "I": I, "I_sd": I_sd, "sound": sound_mean, "sound_sd": sound_sd,
            "raw_drop": raw_drop, "R": np.nan, "R_sd": np.nan,
            "note": "no species information on clean audio",
        }
    per_fold = np.array([robustness(f, s) for f, s in zip(focal, sound)], dtype=float)
    return {
        "I": I, "I_sd": I_sd, "sound": sound_mean, "sound_sd": sound_sd,
        "raw_drop": raw_drop, "R": robustness(I, sound_mean), "R_sd": sample_sd(per_fold),
        "note": "",
    }


def self_check():
    """Known rankings. Raises AssertionError if a formula drifts."""
    y = np.array([0, 0, 0, 1, 1, 1])
    assert abs(auroc(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) - 1) < 1e-12
    assert abs(auroc(y, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])) - 0) < 1e-12
    assert abs(auroc(y, np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-12
    # One tie between a positive and a negative: the pair scores 0.5, the other five score 1.
    tied = auroc(np.array([1, 1, 0, 0]), np.array([0.2, 0.9, 0.2, 0.1]))
    assert abs(tied - (1 + 1 + 0.5 + 1) / 4) < 1e-12
    assert np.isnan(auroc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])))

    perfect = average_precision(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1]))
    assert abs(perfect - 1) < 1e-12
    # Positives ranked 2nd and 3rd. Precisions 1/2 and 2/3.
    mixed = average_precision(np.array([0, 1, 1, 0]), np.array([0.9, 0.8, 0.4, 0.1]))
    assert abs(mixed - (0.5 + 2 / 3) / 2) < 1e-12
    assert np.isnan(average_precision(np.array([0, 0, 0]), np.array([0.2, 0.1, 0.0])))

    strong = family_summary([0.90] * 10, [0.70] * 10)
    assert abs(strong["I"] - 0.90) < 1e-12 and abs(strong["R"] - 0.5) < 1e-12
    assert strong["note"] == "" and strong["R_sd"] == 0
    weak = family_summary([0.55] * 10, [0.52] * 10)
    assert abs(weak["R"] - 0.40) < 1e-12
    none = family_summary([0.50, 0.51, 0.49, 0.50], [0.50, 0.50, 0.50, 0.50])
    assert none["note"] == "no species information on clean audio"
    assert np.isnan(none["R"])

    graded = graded_species(np.array([1, 0, 5, 2]), np.array([5, 9, 4, 10]))
    assert list(graded) == [0, 3]
    assert macro(np.array([0.2, 0.4, np.nan, 0.8]), graded) == 0.5
    assert np.isnan(macro(np.array([np.nan, np.nan]), []))


if __name__ == "__main__":
    self_check()
    print("metrics self-check passed")
