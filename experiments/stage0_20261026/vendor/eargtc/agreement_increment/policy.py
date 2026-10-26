"""Validation-only model and acceptance selection for the agreement increment."""
from fractions import Fraction
import hashlib
import numpy as np

FRACTIONS = (.01, .025, .05, .10, .15, .20, .25, .35, .50, .75, 1.)


def _arrays(score, correct, ids):
    score = np.asarray(score, dtype=float)
    ids = np.asarray(ids, dtype=np.int64)
    if score.ndim != 1 or ids.shape != score.shape or not np.isfinite(score).all():
        raise ValueError('finite scores and IDs must be aligned vectors')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate node IDs')
    if correct is None:
        return score, None, ids
    correct = np.asarray(correct, dtype=bool)
    if correct.shape != score.shape:
        raise ValueError('correctness and scores must align')
    return score, correct, ids


def top_mask(score, ids, fraction):
    score, _, ids = _arrays(score, None, ids)
    f = Fraction(str(fraction))
    if not 0 <= f <= 1:
        raise ValueError('fraction outside [0,1]')
    k = (len(ids)*f.numerator + f.denominator-1)//f.denominator
    mask = np.zeros(len(ids), dtype=bool)
    mask[np.lexsort((ids, -score))[:k]] = True
    return mask


def wilson_lower(successes, count, z=1.6448536269514722):
    if count == 0:
        return None
    p = successes/count
    return float((p+z*z/(2*count)-z*np.sqrt(p*(1-p)/count+z*z/(4*count**2)))/(1+z*z/count))


def summarize(correct, mask):
    correct, mask = np.asarray(correct, bool), np.asarray(mask, bool)
    n = int(mask.sum()); good = int(correct[mask].sum())
    return dict(accepted=n, correct=good, wrong=n-good, population=len(mask),
                coverage=n/len(mask) if len(mask) else 0., precision=good/n if n else None,
                wilson_lower=wilson_lower(good, n))


def select_policy(score, correct, ids, min_accepted=20):
    score, correct, ids = _arrays(score, correct, ids)
    rows = [dict(fraction=f, **summarize(correct, top_mask(score, ids, f))) for f in FRACTIONS]
    valid = [r for r in rows if r['accepted'] >= min_accepted]
    if not valid:
        return dict(reject_all=True, fraction=0., accepted=0, precision=None, rows=rows,
                    reason='insufficient_policy_support', min_accepted=min_accepted)
    best = max(valid, key=lambda r: (r['precision'], r['wilson_lower'], r['fraction']))
    return dict(best, reject_all=False, rows=rows, min_accepted=min_accepted,
                reason='policy_precision_then_wilson_lower_then_coverage')


def calibration_folds(ids, dataset, count=5):
    ids = np.asarray(ids, dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate calibration IDs')
    count = min(count, len(ids))
    if count < 2:
        raise ValueError('at least two calibration rows needed')
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        f'{dataset}|agreement-increment-cv|{int(ids[i])}'.encode()).digest())
    folds = np.empty(len(ids), dtype=np.int64)
    folds[order] = np.arange(len(ids)) % count
    return folds


def selection_metrics(score, correct, ids):
    score, correct, ids = _arrays(score, correct, ids)
    prob = np.clip(score, 1e-7, 1-1e-7)
    bce = float(-np.mean(correct*np.log(prob)+(~correct)*np.log1p(-prob)))
    brier = float(np.mean((score-correct)**2))
    points = [summarize(correct, top_mask(score, ids, f)) for f in (.05, .10, .20)]
    points = [p for p in points if p['accepted'] >= 10]
    order = np.lexsort((ids, -score))
    aurc = float(np.mean(np.cumsum(~correct[order])/np.arange(1, len(ids)+1)))
    from sklearn.metrics import roc_auc_score
    return dict(top_precision=float(np.mean([p['precision'] for p in points])) if points else None,
                metric_regime='top_precision' if points else 'bce_fallback_small_calibration',
                bce=bce, brier=brier, aurc=aurc,
                auroc=float(roc_auc_score(correct, score)) if len(np.unique(correct)) == 2 else None,
                ranking_points=points, positive_count=int(correct.sum()), negative_count=int((~correct).sum()))


def selection_key(record):
    m = record['metrics']
    return (-m['top_precision'] if m['top_precision'] is not None else 0., m['bce'], record['id'])
