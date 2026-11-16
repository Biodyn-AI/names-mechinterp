"""Length changes are allowed; stale or incomplete visual reviews are not."""

import tempfile
import unittest
from pathlib import Path

from export_latex import escape
from validation_rules import review_record, validate_abstract, validate_review_record


class ValidationRulesTests(unittest.TestCase):
    def test_simple_abstract_has_dynamic_count_and_preserves_facts(self):
        summary = (
            "Two labels can name the same object. We asked two models about "
            "32 made-up worlds. One test changed 1 of 8 wrong answers."
        )
        self.assertEqual(validate_abstract(summary), len(summary.split()))
        self.assertEqual(escape(summary), summary)
        self.assertIn(r"\cite{jainwallace2019attention,wiegreffepinter2019attention}",
                      escape("Earlier work [6,7].", citations=True))

    def test_abstract_accepts_limit_but_rejects_empty_or_too_long(self):
        self.assertEqual(validate_abstract("word " * 300), 300)
        for invalid in ("", "   ", "word " * 301):
            with self.assertRaises(ValueError):
                validate_abstract(invalid)

    def test_review_counts_can_change_but_cover_must_remain_one_page(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {}
            for name, count in (("Manuscript", 4), ("Cover_letter", 1)):
                artifacts[name] = root / (name + ".bin")
                artifacts[name].write_bytes(name.encode())
                (root / name).mkdir()
                for number in range(1, count + 1):
                    (root / name / f"page-{number}.png").write_bytes(str(number).encode())
            record = review_record(root, artifacts)
            with self.assertRaises(ValueError):
                validate_review_record(root, artifacts, record)
            record["passed"] = True
            self.assertEqual(validate_review_record(root, artifacts, record),
                             {"Manuscript": 4, "Cover_letter": 1})
            (root / "Cover_letter/page-2.png").write_bytes(b"second")
            with self.assertRaises(ValueError):
                review_record(root, artifacts)

    def test_review_is_invalidated_by_source_or_page_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"original")
            (root / "Manuscript").mkdir()
            page = root / "Manuscript/page-1.png"
            page.write_bytes(b"pixels")
            artifacts = {"Manuscript": source}
            record = review_record(root, artifacts)
            record["passed"] = True
            source.write_bytes(b"revised")
            with self.assertRaises(ValueError):
                validate_review_record(root, artifacts, record)
            source.write_bytes(b"original")
            page.write_bytes(b"changed pixels")
            with self.assertRaises(ValueError):
                validate_review_record(root, artifacts, record)


if __name__ == "__main__":
    unittest.main()
