"""Resampler zoo for the in-DynED balancing-module ablation.

Each mode takes the per-class sliding-window data (X, y) and returns a class-balanced training
batch at ~n_train per class, so the ONLY thing that varies versus LSH-RHP is *how* the balanced
batch is chosen/generated -- the rest of the LSH-DynED pipeline is untouched. This isolates the
selection mechanism (the manuscript's claim is that LSH-RHP ~ random; this tests whether *any*
informed/SOTA selector beats random inside DynED).

Modes
  undersampling (majority -> n_train, minority kept; same per-class counts as DynED random):
    random   - RandomUnderSampler                (baseline; matches get_selected_data_for_train_random)
    cluster  - ClusterCentroids (k-means, hard)  (informed-diversity rival to LSH-RHP)
    spe      - Self-Paced Ensemble undersampler  (SOTA, Liu et al. ICDE 2020; genuine imbens code)
    iht      - InstanceHardnessThreshold         (hardness-based, Smith et al. 2014)
  oversampling (random-cap majority -> n_train, then synthesize minority -> n_train):
    smote    - SMOTE                             (classic)
    gsmote   - Geometric-SMOTE                   (SOTA, Douzas & Bacao 2019; faithful reimpl)

Robust: ANY failure falls back to per-class random selection, so one pathological batch never
crashes a stream. Fallbacks are counted in resample_balanced.FALLBACKS for reporting.
"""
import os
import numpy as np
from collections import Counter

UNDERSAMPLERS = {"random", "cluster", "spe", "iht"}
OVERSAMPLERS = {"smote", "gsmote"}
ALL_MODES = UNDERSAMPLERS | OVERSAMPLERS
SEED = 101
FALLBACKS = Counter()   # mode -> count of batches that fell back to random
ERRORS = []             # distinct "mode: ExcType: msg" strings (capped), for diagnosis


def _per_class_random(X, y, n_train):
    out_X, out_y = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        take = min(n_train, len(idx))
        sel = np.random.choice(idx, take, replace=False)
        out_X.append(X[sel])
        out_y.extend([c] * take)
    return np.vstack(out_X), np.asarray(out_y)


def _geometric_smote(X_min, n_gen, k=5, trunc=0.0, deform=0.0):
    """Geometric-SMOTE (Douzas & Bacao 2019): synthesize each point inside a hypersphere whose
    radius is the distance to a random minority neighbour, with truncation + deformation toward
    the center->neighbour axis. trunc=deform=0 -> hypersphere SMOTE (a strict superset of SMOTE)."""
    n, d = X_min.shape
    if n_gen <= 0:
        return np.empty((0, d), dtype=np.float64)
    if n == 1:
        return np.repeat(X_min.astype(np.float64), n_gen, axis=0)
    from sklearn.neighbors import NearestNeighbors
    kk = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(X_min)
    _, nbrs = nn.kneighbors(X_min)
    synth = np.empty((n_gen, d), dtype=np.float64)
    for s in range(n_gen):
        i = np.random.randint(n)
        center = X_min[i].astype(np.float64)
        j = nbrs[i][np.random.randint(1, kk + 1)]
        surface = X_min[j].astype(np.float64)
        radius = np.linalg.norm(surface - center)
        v = np.random.randn(d)
        nv = np.linalg.norm(v)
        v = v / nv if nv > 1e-12 else v
        r = np.random.rand() ** (1.0 / d)
        p = v * r
        e = surface - center
        en = np.linalg.norm(e)
        if en > 1e-12:
            e = e / en
            proj = p @ e
            if proj < trunc:
                p = p - 2 * proj * e
            perp = p - (p @ e) * e
            p = p - deform * perp
        synth[s] = center + radius * p
    return synth


def _undersample(X, y, mode, strat):
    if mode == "cluster":
        from imblearn.under_sampling import ClusterCentroids
        from sklearn.cluster import KMeans
        cc = ClusterCentroids(sampling_strategy=strat, voting="hard",
                              estimator=KMeans(n_init=1, random_state=SEED), random_state=SEED)
        return cc.fit_resample(X, y)
    if mode == "iht":
        from imblearn.under_sampling import InstanceHardnessThreshold
        return InstanceHardnessThreshold(sampling_strategy=strat, cv=3,
                                         random_state=SEED).fit_resample(X, y)
    if mode == "spe":
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from imbens.sampler import SelfPacedUnderSampler
        base = DecisionTreeClassifier(max_depth=8, random_state=SEED)
        classes_ = np.unique(y)
        n_min = int(min(Counter(y.tolist()).values()))
        cv = min(3, n_min)
        if cv >= 2:   # out-of-fold proba -> honest hardness (in-sample overfits -> degenerate weights)
            proba = cross_val_predict(base, X, y, method="predict_proba",
                                      cv=StratifiedKFold(cv, shuffle=True, random_state=SEED))
        else:
            proba = base.fit(X, y).predict_proba(X)
            classes_ = base.classes_
        emap = {c: i for i, c in enumerate(classes_)}
        spe = SelfPacedUnderSampler(sampling_strategy=strat, random_state=SEED)
        # alpha>0 keeps the self-paced tilt while avoiding the 1/hardness=inf degeneracy at alpha=0
        return spe.fit_resample(X, y, y_pred_proba=proba, alpha=0.1,
                                classes_=classes_, encode_map=emap)
    from imblearn.under_sampling import RandomUnderSampler   # mode == "random"
    return RandomUnderSampler(sampling_strategy=strat, random_state=SEED).fit_resample(X, y)


def _oversample(X, y, mode, over):
    counts = Counter(y.tolist())
    if mode == "gsmote":
        out_X, out_y = [X.astype(np.float64)], [y]
        for c, target in over.items():
            idx = np.where(y == c)[0]
            synth = _geometric_smote(X[idx], target - len(idx))
            if len(synth):
                out_X.append(synth)
                out_y.append(np.full(len(synth), c))
        return np.vstack(out_X), np.concatenate(out_y)
    kmin = min(counts[c] for c in over)            # mode == "smote"
    if kmin < 2:
        from imblearn.over_sampling import RandomOverSampler
        return RandomOverSampler(sampling_strategy=over, random_state=SEED).fit_resample(X, y)
    from imblearn.over_sampling import SMOTE
    return SMOTE(sampling_strategy=over, k_neighbors=max(1, min(5, kmin - 1)),
                 random_state=SEED).fit_resample(X, y)


def resample_balanced(X, y, n_train, mode):
    """Return (X_bal, y_bal) ~ n_train per class. Falls back to per-class random on any failure.

    NaN handling: some streams (e.g. activity) carry missing values; faiss/random tolerate them but
    every sklearn-based sampler rejects NaN. We mean-impute (column mean, 0 if a column is all-NaN)
    so the samplers actually run rather than silently degrading to the random fallback."""
    X = np.array(X, dtype=np.float64)   # copy: we may impute in place
    y = np.asarray(y)
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        nan_idx = np.where(np.isnan(X))
        X[nan_idx] = np.take(col_mean, nan_idx[1])
    try:
        counts = Counter(y.tolist())
        classes = sorted(counts)
        if mode in UNDERSAMPLERS:
            strat = {c: n_train for c in classes if counts[c] > n_train}
            return (X, y) if not strat else _undersample(X, y, mode, strat)
        if mode in OVERSAMPLERS:
            cap = {c: n_train for c in classes if counts[c] > n_train}
            if cap:
                from imblearn.under_sampling import RandomUnderSampler
                X, y = RandomUnderSampler(sampling_strategy=cap,
                                          random_state=SEED).fit_resample(X, y)
                counts = Counter(y.tolist())
            over = {c: n_train for c in classes if counts[c] < n_train}
            return (X, y) if not over else _oversample(X, y, mode, over)
        return _per_class_random(X, y, n_train)
    except Exception as e:
        FALLBACKS[mode] += 1
        msg = f"{mode}: {type(e).__name__}: {str(e)[:200]}"
        if msg not in ERRORS and len(ERRORS) < 20:
            ERRORS.append(msg)
            if os.environ.get("DYNED_DEBUG_SAMPLER"):
                import sys
                cc = dict(Counter(np.asarray(y).tolist()))
                print(f"FBCAUSE {msg} | Xshape={np.asarray(X).shape} counts={cc}",
                      file=sys.stderr, flush=True)
        return _per_class_random(X, y, n_train)
