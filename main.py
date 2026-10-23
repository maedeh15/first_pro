import json
from decimal import Decimal, localcontext
from fractions import Fraction
from math import floor


def _reject_constant(_):
    return None


def _valid(rec):
    if not isinstance(rec, dict):
        return False
    a = rec.get("annotator")
    l = rec.get("label")
    c = rec.get("confidence")
    if not isinstance(a, str) or not a:
        return False
    if not isinstance(l, str) or not l:
        return False
    if isinstance(c, bool) or not isinstance(c, (int, float, Decimal)):
        return False
    return 0 <= c <= 1


def _round2(x):
    return float(Fraction(floor(x * 100 + Fraction(1, 2)), 100))


def consensus(path: str) -> dict:
    empty = {"label": None, "confidence": 0.0, "annotators": 0, "scores": {}}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f, parse_float=Decimal, parse_constant=_reject_constant)
    except (OSError, ValueError, RecursionError):
        return empty
    if not isinstance(data, list):
        return empty

    votes = {}
    for rec in data:
        if _valid(rec):
            votes[(rec["annotator"], rec["label"])] = rec["confidence"]
    if not votes:
        return empty

    people = {a for a, _ in votes}
    totals = {}
    with localcontext() as ctx:
        ctx.prec = 100000
        for (_, label), c in votes.items():
            totals[label] = totals.get(label, 0) + Decimal(c)
    totals = {label: Fraction(t) for label, t in totals.items()}

    n = len(people)
    scores = {label: s / n for label, s in totals.items()}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    accepted = [label for label, s in ranked if s >= Fraction(1, 2)]

    if accepted:
        name = accepted[0]
        if len(accepted) > 1:
            name += " (" + ", ".join(accepted[1:]) + ")"
    else:
        name = None

    return {
        "label": name,
        "confidence": _round2(ranked[0][1]),
        "annotators": n,
        "scores": {label: _round2(s) for label, s in ranked},
    }
