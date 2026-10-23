import json
from decimal import Decimal, ROUND_HALF_UP, getcontext
from fractions import Fraction

getcontext().prec = 50

THRESHOLD = Fraction(1, 2)
EMPTY = {"label": None, "confidence": 0.0, "annotators": 0, "scores": {}}


def _valid_confidence(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0 <= value <= 1


def _round2(value: Fraction) -> float:
    dec = Decimal(value.numerator) / Decimal(value.denominator)
    return float(dec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def consensus(path: str) -> dict:
    """label اجماعی یه رکورد رو از روی فایل JSON داده‌شده حساب کن و برگردون."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    latest = {}
    annotators = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        who, label, conf = rec.get("annotator"), rec.get("label"), rec.get("confidence")
        if not isinstance(who, str) or not who.strip():
            continue
        if not isinstance(label, str) or not label.strip():
            continue
        if not _valid_confidence(conf):
            continue
        who, label = who.strip(), label.strip()
        annotators.add(who)
        latest[(who, label)] = Fraction(str(conf))

    if not annotators:
        return dict(EMPTY, scores={})

    totals = {}
    for (_, label), conf in latest.items():
        totals[label] = totals.get(label, Fraction(0)) + conf

    n = len(annotators)
    raw = {label: total / n for label, total in totals.items()}
    ranked = sorted(raw, key=lambda k: (-raw[k], k))
    scores = {k: _round2(raw[k]) for k in ranked}

    accepted = [k for k in ranked if raw[k] >= THRESHOLD]
    if not accepted:
        return {"label": None, "confidence": scores[ranked[0]], "annotators": n, "scores": scores}

    label = accepted[0]
    if len(accepted) > 1:
        label += " (" + ", ".join(accepted[1:]) + ")"
    return {"label": label, "confidence": scores[accepted[0]], "annotators": n, "scores": scores}
