"""Package explicit submission inputs; exclude logs, QA images and local paths."""

import hashlib
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
DELIVERABLES = (
    "Manuscript.docx", "Manuscript.pdf", "Manuscript_submission.tex",
    "S1_Appendix.docx", "S1_Appendix.pdf", "Cover_letter.docx", "Cover_letter.pdf",
    "Submission_checklist.docx", "Submission_checklist.pdf",
    "Fig1.tif", "Fig2.tif", "S1_Data.zip", "README.md",
)


def write_archive(destination, entries):
    """Fix ZIP metadata so unchanged input bytes produce unchanged archives."""
    entries = list(entries)
    for source, name in entries:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Archive paths must be relative and remain inside its root")
        if not Path(source).is_file():
            raise FileNotFoundError(source)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for source, name in sorted(entries, key=lambda entry: entry[1]):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, Path(source).read_bytes(), compresslevel=9)


def package():
    paper = ROOT / "paper"
    latex_sources = [(source, source.relative_to(paper).as_posix()) for source in (
        paper / "manuscript.tex", paper / "appendix.tex", paper / "cover_letter.tex",
        paper / "references.bib", paper / "vendor/plos2025.bst",
        paper / "vendor/plos_latex_template.tex", paper / "vendor/NOTICE.md",
        paper / "LATEX_BUILD.md",
    )]
    latex_zip = SUBMISSION / "LaTeX_sources.zip"
    write_archive(latex_zip, latex_sources)
    entries = [(SUBMISSION / name, name) for name in DELIVERABLES]
    entries.append((latex_zip, latex_zip.name))
    checksums = SUBMISSION / "SHA256SUMS.txt"
    checksums.write_text("".join(
        hashlib.sha256(source.read_bytes()).hexdigest() + "  " + name + "\n"
        for source, name in sorted(entries, key=lambda entry: entry[1])
    ), encoding="utf-8")
    entries.append((checksums, checksums.name))
    archive = SUBMISSION / "PLOS_ONE_submission_package.zip"
    write_archive(archive, entries)
    print(f"Created {archive.name}: {archive.stat().st_size:,} bytes, {len(entries)} files")


if __name__ == "__main__":
    package()
