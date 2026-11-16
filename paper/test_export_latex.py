"""Model-free tests for escaping and the manuscript's frozen citation map."""

import unittest

from export_latex import escape, citation_keys, tex_table
from publication_content import availability_statement, release_metadata


class LatexExportTests(unittest.TestCase):
    def test_special_characters_and_unicode_are_valid_tex(self):
        escaped = escape("A_1 & 95% → B; Tübingen ≥ 2 × 3")
        self.assertIn(r"A\_1 \& 95\%", escaped)
        self.assertIn(r"$\rightarrow$", escaped)
        self.assertIn(r"$\geq$", escaped)

    def test_only_numeric_reference_groups_become_citations(self):
        text = escape("Prior work [1,6,7,8] differs from [ALIAS_END].", citations=True)
        self.assertIn(r"\cite{yang2026import,jainwallace2019attention,wiegreffepinter2019attention,zhangnanda2024patching}", text)
        self.assertIn(r"[ALIAS\_END]", text)
        self.assertEqual(len(citation_keys()), 11)

    def test_tables_have_repeat_headers_and_escape_cells(self):
        rendered = tex_table(["Item", "Value"], [["A_1", "10%"]])
        self.assertIn(r"\endhead", rendered)
        self.assertIn(r"A\_1", rendered)
        self.assertIn(r"10\%", rendered)

    def test_repository_link_has_no_sentence_punctuation(self):
        url = release_metadata()["repository"]
        rendered = escape(availability_statement())
        self.assertIn(r"\url{" + url + "}", rendered)
        self.assertNotIn(r"\url{" + url + ".}", rendered)


if __name__ == "__main__":
    unittest.main()
