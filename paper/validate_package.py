"""Structural checks supplement, rather than replace, rendered page inspection."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.oxml.ns import qn
from PIL import Image
from validation_rules import validate_abstract, validate_review_record

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "submission"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--qa-word", type=Path, default=ROOT / "qa/release_word")
parser.add_argument("--qa-pdf", type=Path, default=ROOT / "qa/release_latex")
options = parser.parse_args()
paper = json.loads((ROOT / "manuscript.json").read_text())
literature = json.loads((ROOT / "literature.json").read_text())
# These numbers are authored into the prose; changing bibliography order alone
# would silently attach claims to the wrong papers.
assert [entry["id"] for entry in literature["references"]] == [
    "yang2026import", "gurarieh2026mixing", "ohdemberg2026retrieval",
    "dai2026cell", "huang2024ravel", "jainwallace2019attention",
    "wiegreffepinter2019attention", "zhangnanda2024patching", "geiger2025causal",
]
doc = Document(OUT / "Manuscript.docx")
text = "\n".join(p.text for p in doc.paragraphs)
for reference_number, entry in enumerate(literature["references"], 1):
    assert any(
        paragraph.text.startswith(f"{reference_number}. ")
        and entry["title"] in paragraph.text
        for paragraph in doc.paragraphs
    ), f"Reference {reference_number} does not match the manuscript citation map"
abstract_words = validate_abstract(paper["abstract"])
assert paper["abstract"] in text, "Word manuscript does not contain the current abstract"
assert len(paper["title"]) <= 250 and len(paper["short_title"]) <= 100
assert "Ihor Kendiukhov" in text and "0000-0001-5342-1499" in text
assert "ihor.kendiukhov@student.uni-tuebingen.de" in text
assert "The author declares no competing interests." in text
assert "funding" not in text.casefold()
assert len(doc.tables) == 3
numbering = doc.sections[0]._sectPr.find(qn("w:lnNumType"))
assert numbering.get(qn("w:restart")) == "continuous"
assert doc.styles["Normal"].paragraph_format.line_spacing == 2
assert abs(doc.sections[0].page_width.inches - 8.5) < 0.01
assert abs(doc.sections[0].page_height.inches - 11) < 0.01
with ZipFile(OUT / "Manuscript.docx") as archive:
    assert not any(name.startswith("word/media/") for name in archive.namelist())
    assert "word/footnotes.xml" not in archive.namelist()
for tab in doc.tables:
    assert tab.rows[0]._tr.get_or_add_trPr().find(qn("w:tblHeader")) is not None
for index, paragraph in enumerate(doc.paragraphs):
    if re.search(r"Fig [12] displays|Fig [12] shows", paragraph.text):
        assert doc.paragraphs[index+1].text.startswith("Fig ")
for filename in ("Fig1.tif", "Fig2.tif"):
    with Image.open(OUT / filename) as image:
        assert image.mode == "RGB" and image.n_frames == 1
        assert image.tag_v2[259] == 5  # TIFF LZW compression.
        assert all(abs(value-600) < 1 for value in image.info["dpi"])
        assert image.width/600 <= 7.5 and image.height/600 <= 8.75
    assert (OUT / filename).stat().st_size < 10_000_000
with ZipFile(OUT / "S1_Data.zip") as data:
    assert len(json.loads(data.read("prompts.json"))) == 3072
    assert len(json.loads(data.read("worlds.json"))) == 32
    assert len(json.loads(data.read("qwen_generations.json"))) == 3072
    assert len(json.loads(data.read("phi_generations.json"))) == 3072
    scope = json.loads(data.read("package_scope.json"))
    assert scope["reserved_confirmation_outcomes_included"] is False
assert (OUT / "S1_Data.zip").stat().st_size < 20_000_000
word_review = options.qa_word / "review_manifest.json"
pdf_review = options.qa_pdf / "review_manifest.json"
pages = validate_review_record(options.qa_word, {
    name: OUT / (name + ".docx") for name in
    ("Manuscript", "S1_Appendix", "Cover_letter", "Submission_checklist")
}, json.loads(word_review.read_text()))
pdf_pages = validate_review_record(options.qa_pdf, {
    name: OUT / (name + ".pdf") for name in
    ("Manuscript", "S1_Appendix", "Cover_letter")
}, json.loads(pdf_review.read_text()))
# Renderer/compiler logs and auxiliary state are not publication deliverables.
deliverable_suffixes = {".docx", ".pdf", ".tif", ".zip", ".tex", ".md"}
report = {
    "structural_checks_passed": True, "pages": pages, "pdf_pages": pdf_pages,
    "abstract_words": abstract_words,
    "reviewed_manifests_sha256": {
        "word": hashlib.sha256(word_review.read_bytes()).hexdigest(),
        "pdf": hashlib.sha256(pdf_review.read_bytes()).hexdigest(),
    },
    "not_submission_ready_until": ["Verify public versioned release and journal data-policy sufficiency", "Author review and approval of science, AI disclosure, contributions and remaining cover fields"],
    "file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in OUT.iterdir()
                    if p.is_file() and p.suffix in deliverable_suffixes},
}
(options.qa_word/"validation_report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
