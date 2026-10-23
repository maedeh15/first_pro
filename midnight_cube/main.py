import json
import re
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

_TS_RE = re.compile(
    r"^([0-9]{4})-([0-9]{2})-([0-9]{2})"
    r"T([0-9]{2}):([0-9]{2}):([0-9]{2})"
    r"(Z|[+-][0-9]{2}:[0-9]{2})$"
)
_OFF_RE = re.compile(r"^([+-])([0-9]{2}):([0-9]{2})$")


def _is_leap(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _days_in_month(year, month):
    lengths = [31, 29 if _is_leap(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return lengths[month - 1]


def _parse_offset(text):
    m = _OFF_RE.match(text)
    if not m:
        return None
    sign, hh, mm = m.group(1), int(m.group(2)), int(m.group(3))
    if hh > 23 or mm > 59:
        return None
    delta = timedelta(hours=hh, minutes=mm)
    return -delta if sign == "-" else delta


def _parse_timestamp(text):
    """Return the UTC instant (naive datetime) for a strict ISO-ish timestamp, or None."""
    m = _TS_RE.match(text)
    if not m:
        return None
    year, month, day, hour, minute, second = (int(g) for g in m.groups()[:6])
    if month < 1 or month > 12:
        return None
    if day < 1 or day > _days_in_month(year, month):
        return None
    if hour > 23 or minute > 59 or second > 59:
        return None
    tail = m.group(7)
    offset = timedelta(0) if tail == "Z" else _parse_offset(tail)
    if offset is None:
        return None
    try:
        local_dt = datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None
    return local_dt - offset


def _validate_record(obj, products):
    if not isinstance(obj, dict):
        return None
    user_id = obj.get("user_id")
    product = obj.get("product")
    event = obj.get("event")
    timestamp = obj.get("timestamp")
    if isinstance(user_id, bool) or not isinstance(user_id, int):
        return None
    if not isinstance(product, str) or product not in products:
        return None
    if not isinstance(event, str) or not event:
        return None
    if not isinstance(timestamp, str):
        return None
    utc_instant = _parse_timestamp(timestamp)
    if utc_instant is None:
        return None
    return user_id, product, event, utc_instant


def _round4(fraction):
    return float(fraction.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _empty_group():
    return {"sessions": 0, "events": 0, "converted": 0, "users": set()}


def build_report(events_path: str, config: dict) -> dict:
    products = config["funnels"]
    tz_offset = _parse_offset(config["tz_offset"])
    gap_limit = timedelta(minutes=config["session_gap_minutes"])
    date_from = date.fromisoformat(config["date_from"])
    date_to = date.fromisoformat(config["date_to"])

    lines = 0
    invalid = 0
    duplicates = 0
    seen = set()
    # per (user_id, product): list of (utc_instant, event) in file order
    by_user_product = {}

    with open(events_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").rstrip("\r")
            lines += 1
            try:
                obj = json.loads(line)
            except ValueError:
                invalid += 1
                continue
            record = _validate_record(obj, products)
            if record is None:
                invalid += 1
                continue
            user_id, product, event, utc_instant = record
            key = (user_id, product, event, utc_instant)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            by_user_product.setdefault((user_id, product), []).append((utc_instant, event))

    # group stats keyed by (local_date_or_star, product_or_star)
    groups = {}
    all_dates = set()
    all_products = set()

    def add_to_group(d, p, user_id, events_count, converted):
        g = groups.setdefault((d, p), _empty_group())
        g["sessions"] += 1
        g["events"] += events_count
        g["converted"] += 1 if converted else 0
        g["users"].add(user_id)

    for (user_id, product), events in by_user_product.items():
        events.sort(key=lambda e: e[0])
        funnel_steps = products[product]

        session = [events[0]]
        for prev, cur in zip(events, events[1:]):
            if cur[0] - prev[0] > gap_limit:
                _flush_session(session, user_id, product, funnel_steps, tz_offset,
                                date_from, date_to, add_to_group, all_dates, all_products)
                session = [cur]
            else:
                session.append(cur)
        _flush_session(session, user_id, product, funnel_steps, tz_offset,
                        date_from, date_to, add_to_group, all_dates, all_products)

    rows = []
    for d in sorted(all_dates):
        for p in sorted(all_products):
            g = groups.get((d, p))
            if g:
                rows.append(_make_row(d, p, g))
        g_star = groups.get((d, "*"))
        if g_star:
            rows.append(_make_row(d, "*", g_star))
    for p in sorted(all_products):
        g = groups.get(("*", p))
        if g:
            rows.append(_make_row("*", p, g))
    g_total = groups.get(("*", "*"))
    if g_total:
        rows.append(_make_row("*", "*", g_total))

    return {
        "rows": rows,
        "quality": {"lines": lines, "invalid": invalid, "duplicates": duplicates},
    }


def _flush_session(session, user_id, product, funnel_steps, tz_offset, date_from, date_to,
                    add_to_group, all_dates, all_products):
    first_utc = session[0][0]
    local_date = (first_utc + tz_offset).date()
    if local_date < date_from or local_date > date_to:
        return

    converted = _check_funnel([e for _, e in session], funnel_steps)
    events_count = len(session)
    date_str = local_date.isoformat()

    all_dates.add(date_str)
    all_products.add(product)

    add_to_group(date_str, product, user_id, events_count, converted)
    add_to_group(date_str, "*", user_id, events_count, converted)
    add_to_group("*", product, user_id, events_count, converted)
    add_to_group("*", "*", user_id, events_count, converted)


def _check_funnel(event_names, funnel_steps):
    idx = 0
    for name in event_names:
        if idx < len(funnel_steps) and name == funnel_steps[idx]:
            idx += 1
            if idx == len(funnel_steps):
                return True
    return idx == len(funnel_steps)


def _make_row(d, p, g):
    sessions = g["sessions"]
    converted = g["converted"]
    rate = _round4(Decimal(converted) / Decimal(sessions)) if sessions else 0.0
    return {
        "date": d,
        "product": p,
        "users": len(g["users"]),
        "sessions": sessions,
        "events": g["events"],
        "converted": converted,
        "conversion_rate": rate,
    }
