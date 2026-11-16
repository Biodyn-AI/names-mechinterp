"""Small publication limits and hash-bound records of completed visual review."""

import hashlib
import re
from pathlib import Path


def validate_abstract(text):
    count = len(text.split())
    if not 1 <= count <= 300:
        raise ValueError(f"Abstract must contain 1–300 words, found {count}")
    return count


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review_record(qa_directory, artifacts):
    """Create an UNAPPROVED inventory; set passed only after all-page inspection.

    A review record is an explicit attestation, not automated proof of quality.
    Its hashes prevent reusing that attestation after an artifact/page changes.
    """
    documents = {}
    for name, artifact in artifacts.items():
        pages = sorted((Path(qa_directory) / name).glob("page-*.png"))
        numbers = sorted(int(re.fullmatch(r"page-(\d+)\.png", p.name)[1]) for p in pages)
        if not pages or numbers != list(range(1, len(pages) + 1)):
            raise ValueError(f"{name}: rendered pages must be contiguous from 1")
        if name == "Cover_letter" and len(pages) != 1:
            raise ValueError("Cover letter must remain exactly one page")
        documents[name] = {
            "artifact_sha256": _sha256(artifact),
            "reviewed_pages": {p.name: _sha256(p) for p in pages},
        }
    return {"passed": False, "documents": documents}


def validate_review_record(qa_directory, artifacts, record):
    if record.get("passed") is not True:
        raise ValueError("A passed all-page visual review is required")
    current = review_record(qa_directory, artifacts)
    if record.get("documents") != current["documents"]:
        raise ValueError("Visual review is stale or does not cover every current page")
    return {name: len(item["reviewed_pages"])
            for name, item in current["documents"].items()}
