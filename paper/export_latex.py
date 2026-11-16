"""Export editable PLOS-template LaTeX and optionally compile the three PDFs.

JSON files are the shared Word/LaTeX content sources. Exporting overwrites the
generated .tex/.bib files; those files can also be edited and compiled directly.
No model inference, network access or journal submission occurs here.
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from docx import Document

from publication_content import load_manuscript
from validation_rules import validate_abstract

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "submission"


def read(name):
    return json.loads((HERE / name).read_text())


def citation_keys():
    return [r["id"] for r in read("literature.json")["references"]] + [
        "qwenmodel", "phimodel"
    ]


def escape(value, citations=False):
    """Escape prose, preserving only explicitly recognized citations and URLs."""
    symbols = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "→": r"$\rightarrow$", "≥": r"$\geq$", "≤": r"$\leq$",
        "×": r"$\times$", "−": "-", "–": "--", "—": "---",
        "\n": r"\newline "
    }
    tokens = re.split(r"(https?://[^\s\])]+|\[\d+(?:,\s*\d+)*\])", str(value))
    rendered = []
    for token in tokens:
        if token.startswith(("http://", "https://")):
            rendered.append(r"\url{" + token + "}")
        elif citations and re.fullmatch(r"\[\d+(?:,\s*\d+)*\]", token):
            indices = [int(n) - 1 for n in re.findall(r"\d+", token)]
            rendered.append(r"\cite{" + ",".join(citation_keys()[n] for n in indices) + "}")
        else:
            rendered.append("".join(symbols.get(c, c) for c in token))
    return "".join(rendered)


def tex_table(headers, rows, title="", note="", widths=None):
    """Cell-based, multipage tables with repeating headers and no raster text."""
    n = len(headers)
    fractions = widths or [1 / n] * n
    columns = "|" + "|".join(
        r">{\raggedright\arraybackslash}p{\dimexpr" + f"{f:.4f}" +
        r"\linewidth-2\tabcolsep-2\arrayrulewidth\relax}"
        for f in fractions
    ) + "|"
    header = " & ".join(r"\textbf{" + escape(h) + "}" for h in headers) + r"\\\hline"
    lines = [r"\begingroup\small\setstretch{1.15}", r"\begin{longtable}{" + columns + "}"]
    if title:
        lines.append(r"\multicolumn{" + str(n) + r"}{@{}p{\linewidth}@{}}{\textbf{" + escape(title) + r"}}\\[5pt]")
    lines += [r"\hline", header, r"\endfirsthead", r"\hline", header, r"\endhead"]
    for row in rows:
        lines.append(" & ".join(escape(cell) for cell in row) + r"\\\hline")
    lines += [r"\end{longtable}"]
    if note:
        lines.append(escape(note) + r"\par")
    lines.append(r"\endgroup")
    return "\n".join(lines)


def preamble():
    template = (HERE / "vendor/plos_latex_template.tex").read_text()
    text = template[template.index(r"\documentclass"):template.index(r"\begin{document}")]
    text = text.replace(r"%\usepackage{setspace}", r"\usepackage{setspace}")
    text = text.replace(r"%\doublespacing", r"\doublespacing")
    text = text.replace(r"\bibliographystyle{plos2025}", r"\bibliographystyle{vendor/plos2025}")
    text = text.replace(r"\lfoot{\today}", r"\lfoot{16 November 2026}")
    generated = (
        "% Generated from the official PLOS template version 3.8, April 2026.\n"
        + text + "\n" + r"\usepackage[T1]{fontenc}" + "\n"
        + r"\usepackage[utf8]{inputenc}" + "\n"
        + r"\usepackage{longtable}" + "\n"
        + r"\hypersetup{hidelinks}" + "\n"
        + r"\setlength{\emergencystretch}{2em}" + "\n"
    )
    return "\n".join(line.rstrip() for line in generated.splitlines()) + "\n"


def bibliography():
    entries = []
    for ref in read("literature.json")["references"]:
        kind = "article" if ref.get("volume") else "inproceedings"
        people = " and ".join(p["family"] + ", " + p["given"] for p in ref["authors"])
        fields = {"author": people, "title": "{" + ref["title"] + "}", "year": ref["year"]}
        fields["journal" if kind == "article" else "booktitle"] = ref["venue"]
        for key in ("volume", "number", "pages", "publisher", "doi", "url"):
            if ref.get(key):
                fields[key] = ref[key]
        entries.append("@" + kind + "{" + ref["id"] + ",\n" + ",\n".join(
            "  " + key + " = {" + str(value).replace("–", "--").replace("&", r"\&") + "}"
            for key, value in fields.items()
        ) + "\n}")
    for key, person, title, year, url in (
        ("qwenmodel", "Qwen Team", "Qwen3-4B-Instruct-2507 model card", 2025,
         "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554"),
        ("phimodel", "Microsoft", "Phi-3.5-mini-instruct model card", 2024,
         "https://huggingface.co/microsoft/Phi-3.5-mini-instruct/tree/2fe192450127e6a83f7441aef6e3ca586c338b77")
    ):
        entries.append("@misc{" + key + ",\n  author = {{" + person + "}},\n  title = {{" + title + "}},\n  year = {" + str(year) + "},\n  howpublished = {Hugging Face},\n  url = {" + url + "},\n  note = {Accessed 15 November 2026}\n}")
    return "\n\n".join(entries) + "\n"


def export():
    manuscript = load_manuscript()
    validate_abstract(manuscript["abstract"])
    author = read("author_details.json")["authors_in_article_order"][0]
    lines = [preamble(), r"\begin{document}", r"\vspace*{0.2in}", r"\begin{flushleft}",
             r"{\Large\textbf{" + escape(manuscript["title"]) + r"}}\par\bigskip",
             escape(author["display_name"]) + r"\textsuperscript{1*}\par\medskip",
             r"\textbf{1} " + escape(author["affiliations"][0]["display"]) + r"\par\medskip",
             "* " + escape(author["email"]) + r"\par",
             "ORCID: " + escape(author["ORCID"]) + r"\par",
             "Short title: " + escape(manuscript["short_title"]), r"\end{flushleft}",
             r"\clearpage", r"\newgeometry{top=0.85in,left=1in,right=1in,footskip=0.75in}",
             r"\fancyheadoffset[L]{0in}\fancyfootoffset[L]{0in}",
             r"\section*{Abstract}", escape(manuscript["abstract"]),
             r"\clearpage", r"\newgeometry{top=0.85in,left=1in,right=1in,footskip=0.75in}",
             r"\fancyheadoffset[L]{0in}\fancyfootoffset[L]{0in}", r"\linenumbers"]
    for block in manuscript["blocks"]:
        kind = block["kind"]
        if kind == "table":
            widths = {"behavior": [.48,.26,.26], "stages": [.20,.25,.28,.27], "decomposition": [.41,.16,.20,.23], "confirmation_repair": [.40,.20,.20,.20]}[block["id"]]
            lines.append(tex_table(block["headers"], block["rows"], block["title"], block["note"], widths))
        elif kind == "references":
            lines.append(r"\nolinenumbers\bibliography{references}")
        elif kind in ("h1", "h2"):
            # BibTeX supplies its own References heading.
            if block["text"] != "References":
                command = "section" if kind == "h1" else "subsection"
                lines.append("\\" + command + "*{" + escape(block["text"]) + "}")
        elif kind == "caption":
            # PLOS initial submission PDF contains legends, never figure images.
            match = re.match(r"Fig (\d+)\. (.*)", block["text"])
            lines.append(r"\begin{figure}[!h]\caption{" + escape(match[2], True) + r"}\label{fig" + match[1] + r"}\end{figure}")
        else:
            lines.append(escape(block["text"], True) + "\n")
    lines.append(r"\end{document}")
    (HERE / "manuscript.tex").write_text("\n\n".join(lines) + "\n")
    (HERE / "references.bib").write_text(bibliography())
    supplement = read("supplement.json")
    lines = [preamble(), r"\begin{document}",
             r"\newgeometry{top=0.85in,left=1in,right=1in,footskip=0.75in}",
             r"\fancyheadoffset[L]{0in}\fancyfootoffset[L]{0in}",
             r"\section*{S1 Appendix}", escape(supplement["title"]), r"\linenumbers"]
    for section in supplement["sections"]:
        heading = re.sub(r"^S(\d+)\.\s*", r"S\1 ", section["heading"])
        heading = re.sub(r"[^\w\s]", "", heading)
        lines.append(r"\section*{" + escape(heading) + "}")
        lines += [escape(p) + "\n" for p in section["paragraphs"]]
        if "table" in section:
            tab = section["table"]
            widths = {2:[.31,.69],3:[.40,.40,.20],4:[.40,.20,.20,.20],5:[.10,.25,.10,.28,.27]}[len(tab["headers"])]
            lines.append(tex_table(tab["headers"], tab["rows"], widths=widths))
    lines.append(r"\end{document}")
    (HERE / "appendix.tex").write_text("\n\n".join(lines) + "\n")
    cover = Document(OUT / "Cover_letter.docx")
    lines = [r"\documentclass[11pt,letterpaper]{article}",
             r"\usepackage[margin=0.85in]{geometry}", r"\usepackage[T1]{fontenc}",
             r"\usepackage[utf8]{inputenc}", r"\usepackage{hyperref,setspace}",
             r"\hypersetup{hidelinks}", r"\setstretch{1.08}", r"\setlength{\parindent}{0pt}",
             r"\setlength{\parskip}{6pt}", r"\begin{document}"]
    for p in cover.paragraphs:
        if p.style.name == "Title":
            lines.append(r"{\Large\textbf{" + escape(p.text) + r"}}\par\medskip")
        else:
            lines.append(escape(p.text) + "\n")
    lines.append(r"\end{document}")
    (HERE / "cover_letter.tex").write_text("\n\n".join(lines) + "\n")


def compile_pdfs(latexmk="latexmk"):
    executable = shutil.which(latexmk)
    if not executable:
        raise RuntimeError("Install TeX Live/MacTeX with latexmk, or pass --latexmk PATH")
    for source, name in (("manuscript", "Manuscript"), ("appendix", "S1_Appendix"), ("cover_letter", "Cover_letter")):
        subprocess.run([executable, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        "-outdir=" + str(OUT), "-jobname=" + name, source + ".tex"], cwd=HERE, check=True)
    # Also supply one cohesive manuscript source with the generated bibliography
    # embedded, as requested by the journal at source-file submission.
    text = (HERE / "manuscript.tex").read_text()
    text = text.replace(r"\nolinenumbers\bibliography{references}",
                        "\\nolinenumbers\n" + (OUT / "Manuscript.bbl").read_text())
    (OUT / "Manuscript_submission.tex").write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--latexmk", default="latexmk")
    options = parser.parse_args()
    export()
    if options.compile:
        compile_pdfs(options.latexmk)
