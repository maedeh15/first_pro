"""Sample tests for consensus. From the project root run:

    python3 -m unittest test.sample_test -v

They only use the sample in sample/ and the empty-list case from the statement; the judge's tests are different.
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import guard  # runs consensus in a separate process with a time limit
except ValueError:  # no "fork" on Windows: call consensus directly, without a time limit
    guard = None

TIMEOUT = 5.0
SAMPLE_INPUT = os.path.join(ROOT, "sample", "labels.json")
SAMPLE_EXPECTED = os.path.join(ROOT, "sample", "expected.json")
EMPTY_RESULT = {"label": None, "confidence": 0.0, "annotators": 0, "scores": {}}


def run_consensus(path):
    if guard is None:
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from main import consensus
        return consensus(path)
    try:
        result, _ = guard.call(ROOT, "consensus", (path,), TIMEOUT)
    except guard.Timeout:
        raise AssertionError("تابع consensus بعد از %s ثانیه هنوز جوابی برنگردانده بود." % TIMEOUT)
    except guard.CallError as exc:
        raise AssertionError("اجرای consensus با خطا تمام شد: %s" % exc)
    return result


class SampleTest(unittest.TestCase):
    _sample = None  # consensus runs once on the sample; every test checks one part of that result

    @classmethod
    def sample_result(cls):
        if cls._sample is None:
            try:
                cls._sample = ("ok", run_consensus(SAMPLE_INPUT))
            except Exception as exc:  # noqa: BLE001 - show the participant's error in every test
                cls._sample = ("fail", str(exc) if isinstance(exc, AssertionError) else "%s: %s" % (type(exc).__name__, exc))
        if cls._sample[0] == "fail":
            raise AssertionError(cls._sample[1])
        return cls._sample[1]

    def setUp(self):
        with open(SAMPLE_EXPECTED, encoding="utf-8") as f:
            self.want = json.load(f)

    def got(self):
        got = self.sample_result()
        self.assertIsInstance(got, dict, "خروجی consensus باید یک دیکشنری باشد، اما %s برگشت." % type(got).__name__)
        return got

    def test_sample_output_equals_expected_file(self):
        got = self.got()
        self.assertEqual(got, self.want, "خروجی برای sample/labels.json باید برابر محتوای sample/expected.json باشد.")

    def test_sample_output_has_exactly_four_keys(self):
        self.assertEqual(sorted(self.got()), ["annotators", "confidence", "label", "scores"],
                         "خروجی باید دقیقاً چهار کلید label، confidence، annotators و scores داشته باشد.")

    def test_sample_output_value_types(self):
        got = self.got()
        self.assertIsInstance(got.get("label"), str, "در نمونه دو برچسب پذیرفته شده‌اند، پس label باید رشته باشد.")
        self.assertIsInstance(got.get("confidence"), float, "confidence باید float باشد.")
        self.assertIs(type(got.get("annotators")), int, "annotators باید int باشد.")
        self.assertIsInstance(got.get("scores"), dict, "scores باید دیکشنری باشد.")
        for label, score in got["scores"].items():
            self.assertIsInstance(score, float, "امتیاز %r در scores باید float باشد." % label)

    def test_sample_counts_six_valid_annotators(self):
        self.assertEqual(self.got().get("annotators"), self.want["annotators"],
                         "در نمونه فقط شش نفر رأی معتبر دارند: رأی a07 با confidence برابر true و رکورد بدون نام "
                         "برچسب‌زن کنار گذاشته می‌شوند.")

    def test_sample_scores_rounded_and_ordered(self):
        got = self.got().get("scores")
        self.assertEqual(got, self.want["scores"], "امتیاز هر برچسب: جمع confidence آخرین رأی‌ها تقسیم بر تعداد "
                                                   "برچسب‌زن‌ها، گردشده با half-up تا دو رقم اعشار.")
        self.assertEqual(list(got), list(self.want["scores"]), "کلیدهای scores باید به ترتیب امتیاز، از زیاد به کم، بیایند.")

    def test_sample_label_lists_accepted_labels(self):
        self.assertEqual(self.got().get("label"), self.want["label"],
                         "هر دو برچسب نمونه پذیرفته شده‌اند؛ اولی بیرون پرانتز و بقیه داخل پرانتز می‌آیند.")

    def test_sample_confidence_is_top_score(self):
        self.assertEqual(self.got().get("confidence"), self.want["confidence"],
                         "confidence برابر امتیاز گردشدهٔ اولین برچسب پس از مرتب‌سازی است.")

    def test_empty_list_gives_fixed_empty_result(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("[]\n")
        try:
            got = run_consensus(f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual(got, EMPTY_RESULT, "برای فهرست خالی، همان دیکشنری ثابتِ صورت سؤال باید برگردد.")


if __name__ == "__main__":
    unittest.main()
