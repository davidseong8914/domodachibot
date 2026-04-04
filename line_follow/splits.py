"""Train / val / test index assignment for steering dataset."""

from __future__ import annotations

import random
from typing import List, Sequence, Set, Tuple


def split_random(n: int, val_frac: float, test_frac: float, seed: int) -> Tuple[Set[int], Set[int], Set[int]]:
    """Disjoint index sets; train gets the remainder. If test_frac <= 0, test set is empty (train/val only)."""
    if n < 2:
        raise ValueError("Need at least 2 samples")
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    vf = max(0.0, min(val_frac, 0.49))
    tf = max(0.0, min(test_frac, 0.49))
    if tf <= 0.0:
        n_val = max(1, int(round(n * vf))) if vf > 0 else max(1, int(round(n * 0.15)))
        if n_val >= n:
            n_val = max(1, n // 5)
        val_set = set(idx[:n_val])
        train_set = set(idx[n_val:])
        if not train_set:
            raise ValueError("Random split produced empty train set")
        return train_set, val_set, set()
    if n < 3:
        raise ValueError("Need at least 3 samples for train/val/test split")
    if vf + tf >= 0.98:
        raise ValueError("val_frac + test_frac must leave room for training")
    n_test = max(1, int(round(n * tf)))
    n_val = max(1, int(round(n * vf)))
    if n_test + n_val >= n:
        n_test = max(1, n // 5)
        n_val = max(1, n // 5)
        if n_test + n_val >= n:
            n_val = 1
            n_test = 1
    test_set = set(idx[:n_test])
    val_set = set(idx[n_test : n_test + n_val])
    train_set = set(idx[n_test + n_val :])
    if not train_set:
        raise ValueError("Random split produced empty train set; reduce val/test fractions")
    return train_set, val_set, test_set


def split_by_path_substrings(
    image_paths: Sequence[str],
    clip_test: Sequence[str],
    clip_val: Sequence[str],
) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    Assign each row index by first match: any of clip_test substrings in path -> test,
    then any of clip_val -> val, else train.
    """
    train_set: Set[int] = set()
    val_set: Set[int] = set()
    test_set: Set[int] = set()
    for i, p in enumerate(image_paths):
        if any(s in p for s in clip_test):
            test_set.add(i)
        elif any(s in p for s in clip_val):
            val_set.add(i)
        else:
            train_set.add(i)
    if not train_set:
        raise ValueError(
            "Clip split produced empty train set. Add more clips to labels, "
            "use different --clip-test/--clip-val, or use --split random."
        )
    if not val_set:
        raise ValueError("Clip split produced empty val set.")
    if not test_set:
        raise ValueError("Clip split produced empty test set.")
    return train_set, val_set, test_set
