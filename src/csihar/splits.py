"""Development/validation split construction (capture-level)."""
import random
from collections import defaultdict


def room_b_validation(files_by_label, per_class=25, seed=42):
    """Pick the Room-B validation captures.

    For each activity (processed in sorted label order) the candidate file
    names are sorted lexicographically and `per_class` are drawn with
    `random.Random(seed).sample`, using a fresh RNG seeded with `seed` per
    class so that one class's draw does not depend on another's size.
    Returns the set of selected file names.
    """
    chosen = set()
    for label in sorted(files_by_label):
        candidates = sorted(files_by_label[label])
        rng = random.Random(seed)
        chosen.update(rng.sample(candidates, per_class))
    return chosen


def group_by_label(items):
    """items: iterable of (name, label) -> {label: [names]}"""
    out = defaultdict(list)
    for name, label in items:
        out[label].append(name)
    return out
