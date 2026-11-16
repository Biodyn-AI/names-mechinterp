# Draft submission package

These are research submission artifacts, not evidence of journal acceptance.
The numerical study and development/confirmation limits are summarized in the
[repository README](https://github.com/Biodyn-AI/names-mechinterp#readme) and
[study scope](https://github.com/Biodyn-AI/names-mechinterp/blob/main/docs/study.md).

| File | Purpose |
|---|---|
| [Manuscript.pdf](Manuscript.pdf) | Main PLOS-style LaTeX PDF; figures are separate |
| [Manuscript.docx](Manuscript.docx) | Editable Word manuscript |
| [Manuscript_submission.tex](Manuscript_submission.tex) | Submission-oriented main TeX source with inline bibliography |
| [S1_Appendix.pdf](S1_Appendix.pdf), [S1_Appendix.docx](S1_Appendix.docx) | Extended methods, cases, criteria, and limitations |
| [Fig1.tif](Fig1.tif), [Fig2.tif](Fig2.tif) | Separate research figures |
| [S1_Data.zip](S1_Data.zip) | Processed supporting data, not the full native tensor archive |
| [Cover_letter.pdf](Cover_letter.pdf), [Cover_letter.docx](Cover_letter.docx) | Draft cover letter |
| [Submission_checklist.docx](Submission_checklist.docx) | Human checks still required before journal submission |
| [LaTeX_sources.zip](LaTeX_sources.zip) | Portable TeX sources, bibliography, and vendor notices |
| [Submission_checklist.pdf](Submission_checklist.pdf) | Read-only copy of the submission checklist |
| [PLOS_ONE_submission_package.zip](PLOS_ONE_submission_package.zip) | Download all submission documents together |
| [SHA256SUMS.txt](SHA256SUMS.txt) | SHA-256 checksums of the individual submission files |

The maintained sources are in [`paper/`](https://github.com/Biodyn-AI/names-mechinterp/tree/main/paper): manuscript/appendix/cover-letter
TeX, `references.bib`, JSON content, and build scripts. Rebuild Word/figure/data
outputs with `python paper/build_submission.py`; rebuild PDFs with
`python paper/export_latex.py --compile` after installing the separate TeX
toolchain. See [build instructions](https://github.com/Biodyn-AI/names-mechinterp/blob/main/docs/reproducing.md#paper-build).

Original manuscript, figures, and processed study data are
[CC BY 4.0](https://github.com/Biodyn-AI/names-mechinterp/blob/main/LICENSE-DATA.md).
Build code is [MIT](https://github.com/Biodyn-AI/names-mechinterp/blob/main/LICENSE).
Vendored journal styles retain their upstream notices. No model weights are
included, and no paper DOI or journal-acceptance status is claimed.
