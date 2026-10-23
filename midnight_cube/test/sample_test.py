"""Sample tests for build_report, built only from the statement's example (sample/).

Run them from the project root:

    python3 -m unittest test.sample_test -v
"""
import json
import os
import sys
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TIMEOUT = 5  # seconds for one call to build_report
SAMPLE = os.path.join(ROOT, "sample")
ROW_KEYS = ["conversion_rate", "converted", "date", "events", "product", "sessions", "users"]


def load_json(name):
    with open(os.path.join(SAMPLE, name), encoding="utf-8") as f:
        return json.load(f)


def sample_lines():
    with open(os.path.join(SAMPLE, "events.jsonl"), encoding="utf-8") as f:
        return f.read().splitlines()


def call_build_report(events_path):
    """Call main.build_report in a background thread, so a hang fails the test instead of freezing the run."""
    box = {}

    def target():
        try:
            from main import build_report
            box["result"] = build_report(events_path, load_json("config.json"))
        except BaseException as exc:  # noqa: BLE001 - report whatever build_report raised
            box["error"] = "%s: %s" % (type(exc).__name__, exc)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(TIMEOUT)
    if worker.is_alive():
        raise AssertionError("build_report بعد از %d ثانیه هنوز جواب نداده بود." % TIMEOUT)
    if "error" in box:
        raise AssertionError("اجرای build_report با خطا تمام شد: " + box["error"])
    return box["result"]


def report_for_lines(lines):
    """build_report on a file made of the given lines of sample/events.jsonl."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write("".join(line + "\n" for line in lines))
    try:
        return call_build_report(f.name)
    finally:
        os.unlink(f.name)


def row(date, product, users, sessions, events, converted, rate):
    return {"date": date, "product": product, "users": users, "sessions": sessions, "events": events,
            "converted": converted, "conversion_rate": rate}


class SampleTest(unittest.TestCase):
    _sample = None  # (report, error message) of the run on sample/events.jsonl, shared by the tests

    def sample_report(self):
        if SampleTest._sample is None:
            try:
                SampleTest._sample = (call_build_report(os.path.join(SAMPLE, "events.jsonl")), None)
            except AssertionError as exc:
                SampleTest._sample = (None, str(exc))
        report, error = SampleTest._sample
        if error:
            self.fail(error)
        self.assertIsInstance(report, dict, "خروجی build_report باید یک dict باشد.")
        return report

    def test_sample_report_equals_report_json(self):
        got, want = self.sample_report(), load_json("report.json")
        self.assertEqual(got.get("quality"), want["quality"], "شمارنده‌های quality با sample/report.json فرق دارند.")
        for i, (a, b) in enumerate(zip(got.get("rows", []), want["rows"])):
            self.assertEqual(a, b, "سطر %d با sample/report.json فرق دارد." % (i + 1))
        self.assertEqual(len(got.get("rows", [])), len(want["rows"]), "تعداد سطرها با sample/report.json فرق دارد.")

    def test_sample_report_has_only_rows_and_quality(self):
        got = self.sample_report()
        self.assertEqual(sorted(got), ["quality", "rows"], "خروجی باید دقیقاً دو کلید rows و quality داشته باشد.")
        self.assertIsInstance(got["rows"], list, "مقدار rows باید list باشد.")
        self.assertEqual(sorted(got["quality"]), ["duplicates", "invalid", "lines"],
                         "کلیدهای quality باید دقیقاً lines و invalid و duplicates باشند.")

    def test_sample_rows_have_seven_typed_fields(self):
        got = self.sample_report()
        self.assertTrue(got["rows"], "گزارش نمونه شش سطر دارد؛ rows خالی است.")
        for r in got["rows"]:
            self.assertEqual(sorted(r), ROW_KEYS, "هر سطر باید دقیقاً این کلیدها را داشته باشد: " + ", ".join(ROW_KEYS))
            for key in ("users", "sessions", "events", "converted"):
                self.assertIs(type(r[key]), int, "مقدار %s باید int باشد." % key)
            self.assertIsInstance(r["conversion_rate"], float, "مقدار conversion_rate باید float باشد (حتی 0.0).")

    def test_sample_quality_counts_lines_invalid_and_duplicates(self):
        got = self.sample_report()
        self.assertEqual(got["quality"], {"lines": 11, "invalid": 2, "duplicates": 1},
                         "نمونه ۱۱ خط دارد: خط دهم و خط آخر نامعتبرند و خط چهارم تکرار خط سوم است.")

    def test_sample_rows_sorted_with_star_last(self):
        got = self.sample_report()
        keys = [(r.get("date"), r.get("product")) for r in got["rows"]]
        want = [("2026-01-01", "editor"), ("2026-01-01", "forms"), ("2026-01-01", "*"),
                ("*", "editor"), ("*", "forms"), ("*", "*")]
        self.assertEqual(keys, want, "سطرها اول بر اساس date و بعد product مرتب می‌شوند و «*» بعد از بقیه می‌آید.")

    def test_sample_total_row_counts_each_user_once(self):
        got = self.sample_report()
        total = [r for r in got["rows"] if r.get("date") == "*" and r.get("product") == "*"]
        self.assertEqual(total, [row("*", "*", 3, 4, 8, 1, 0.25)],
                         "سطر («*»، «*») سه کاربر، چهار نشست، هشت رویداد و یک نشست موفق دارد.")

    def test_sample_user_12_has_two_sessions_one_converted(self):
        got = report_for_lines(sample_lines()[:5])
        self.assertIsInstance(got, dict, "خروجی build_report باید یک dict باشد.")
        rows = [row(d, p, 1, 2, 4, 1, 0.5) for d, p in
                (("2026-01-01", "editor"), ("2026-01-01", "*"), ("*", "editor"), ("*", "*"))]
        self.assertEqual(got.get("quality"), {"lines": 5, "invalid": 0, "duplicates": 1},
                         "پنج خط اول نمونه: یک تکراری و بدون خط نامعتبر.")
        self.assertEqual(got.get("rows"), rows,
                         "کاربر ۱۲ دو نشست دارد (صبح و عصر) و فقط نشست صبح قیف را کامل می‌کند.")

    def test_sample_last_six_lines_have_two_invalid(self):
        got = report_for_lines(sample_lines()[5:])
        self.assertIsInstance(got, dict, "خروجی build_report باید یک dict باشد.")
        rows = [row("2026-01-01", "editor", 1, 1, 2, 0, 0.0), row("2026-01-01", "forms", 1, 1, 2, 0, 0.0),
                row("2026-01-01", "*", 2, 2, 4, 0, 0.0), row("*", "editor", 1, 1, 2, 0, 0.0),
                row("*", "forms", 1, 1, 2, 0, 0.0), row("*", "*", 2, 2, 4, 0, 0.0)]
        self.assertEqual(got.get("quality"), {"lines": 6, "invalid": 2, "duplicates": 0},
                         "شش خط آخر نمونه: respond با فاصله به‌جای T و خط ناقص آخر نامعتبرند.")
        self.assertEqual(got.get("rows"), rows, "کاربر ۱۸ و کاربر ۲۱ هر کدام یک نشست ناموفق دارند.")


if __name__ == "__main__":
    unittest.main()
