# Building the LaTeX documents

The archive includes editable manuscript, appendix, and cover-letter sources,
the bibliography, and the journal's original template and bibliography style.
Install TeX Live or MacTeX with `latexmk`, then run these commands from this folder:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error appendix.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error cover_letter.tex
```

The manuscript follows the official PLOS template. Its PDF contains text, tables,
legends, and references, with no embedded figures. Upload the separate figure
files supplied in the full submission package. No journal submission is performed
by these commands.

In the repository, JSON content in `paper/` is shared between the Word and LaTeX
builders. `python paper/export_latex.py --compile` regenerates the editable TeX
files and the PDFs. Changes made directly to the generated TeX are preserved only
if you compile those files directly instead of rerunning the exporter.

The full package also supplies `Manuscript_submission.tex`, a single main source
with its bibliography embedded. Compile it directly with `pdflatex` twice; it does
not require BibTeX or a separate `.bib` file. The current documents remain drafts
until the author completes the marked confirmations and final scientific review.

Original paper content is licensed under CC BY 4.0; build code is MIT licensed.
Third-party files in `vendor/` retain the notices described in `vendor/NOTICE.md`.
