"""Build the PLOS ONE draft and processed supporting files without model calls."""

import hashlib
import json
import math
import random
import re
import string
import textwrap
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont
from publication_content import availability_statement, load_manuscript, release_metadata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "original"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "submission"
OUT.mkdir(exist_ok=True)
QA = HERE / "qa"
QA.mkdir(exist_ok=True)
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)


def read(path):
    return json.loads(Path(path).read_text())


def sealed(path):
    path = Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == Path(str(path) + ".sha256").read_text().strip(), path
    return read(path)


paper, supplement, literature, evidence, author_record, rules = [read(HERE / name) for name in (
    "manuscript.json", "supplement.json", "literature.json", "evidence_digest.json", "author_details.json", "journal_checklist.json")]
paper = load_manuscript()
release = release_metadata()
author = author_record["authors_in_article_order"][0]
assert author_record["verified_article_author_count"] == 1
assert len(paper["abstract"].split()) <= 300
assert len(paper["title"]) <= 250 and len(paper["short_title"]) <= 100


def document(title, double=True, numbers=True):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin = sec.bottom_margin = Inches(0.8)
    sec.left_margin = sec.right_margin = Inches(0.9)
    for name in ("Normal", "Title", "Heading 1", "Heading 2", "Heading 3", "Caption"):
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.size = Pt(12)
        style.paragraph_format.line_spacing = 2 if double else 1.15
        style.paragraph_format.space_after = Pt(0 if double else 6)
        if style._element.pPr is not None:
            for border in list(style._element.pPr.findall(qn("w:pBdr"))):
                style._element.pPr.remove(border)
    doc.styles["Title"].font.size = Pt(16)
    doc.styles["Title"].font.bold = True
    doc.styles["Title"].paragraph_format.space_after = Pt(14)
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        doc.styles[name].font.bold = True
        doc.styles[name].paragraph_format.space_before = Pt(12)
        doc.styles[name].paragraph_format.space_after = Pt(3)
    if numbers:
        numbering = OxmlElement("w:lnNumType")
        for key, val in {"countBy":"1", "distance":"120", "restart":"continuous"}.items():
            numbering.set(qn("w:" + key), val)
        sec._sectPr.append(numbering)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    doc.core_properties.title = title
    doc.core_properties.author = author["display_name"]
    doc.core_properties.subject = "PLOS ONE research article draft"
    doc.core_properties.comments = ""
    return doc


def paragraph(doc, text, style=None):
    p = doc.add_paragraph(text, style)
    p.paragraph_format.widow_control = True
    return p


def table(doc, title, headers, rows, note="", widths=None):
    if title:
        caption = paragraph(doc, title)
        caption.paragraph_format.keep_with_next = True
        caption.runs[0].bold = True
    tab = doc.add_table(rows=1, cols=len(headers))
    tab.autofit = False
    widths = widths or ([3.0, 1.8, 1.8] if len(headers) == 3 else [6.7 / len(headers)] * len(headers))
    for col, width in zip(tab.columns, widths):
        col.width = Inches(width)
    repeat = OxmlElement("w:tblHeader")
    tab.rows[0]._tr.get_or_add_trPr().append(repeat)
    for index, text in enumerate(headers):
        tab.rows[0].cells[index].text = text
    for values in rows:
        for cell, text in zip(tab.add_row().cells, values):
            cell.text = str(text)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        line = OxmlElement("w:" + edge)
        for key, val in {"val":"single", "sz":"4", "color":"D9D9D9"}.items():
            line.set(qn("w:" + key), val)
        borders.append(line)
    tab._tbl.tblPr.append(borders)
    for row_index, row in enumerate(tab.rows):
        no_split = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(no_split)
        for cell_index, cell in enumerate(row.cells):
            cell.width = Inches(widths[cell_index])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            properties = cell._tc.get_or_add_tcPr()
            margins = OxmlElement("w:tcMar")
            for edge in ("top", "left", "bottom", "right"):
                side = OxmlElement("w:" + edge)
                side.set(qn("w:w"), "75")
                side.set(qn("w:type"), "dxa")
                margins.append(side)
            properties.append(margins)
            if row_index == 0:
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), "EEEEEE")
                properties.append(shading)
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing = 1.25
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.keep_with_next = row_index == 0
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if cell_index == 0 else WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(10.5)
                    run.bold = row_index == 0
    if note:
        p = paragraph(doc, note)
        for run in p.runs:
            run.font.size = Pt(10.5)
    return tab


def initials(person):
    return "".join(part[0] for part in re.findall(r"[A-Za-z]+", person["given"]))


references = []
for index, ref in enumerate(literature["references"], 1):
    people = ref["authors"]
    names = ", ".join(p["family"] + " " + initials(p) for p in people[:6])
    if len(people) > 6:
        names += ", et al"
    if ref.get("volume"):
        publication = f'{ref["venue"]}. {ref["year"]};{ref["volume"]}({ref["number"]}):{ref["pages"]}.'
    else:
        publication = f'In: {ref["venue"]}. {ref["publisher"]}; {ref["year"]}.'
        if ref.get("pages"):
            publication += f' p. {ref["pages"]}.'
    location = "https://doi.org/" + ref["doi"] if ref.get("doi") else ref["url"]
    references.append(f'{index}. {names}. {ref["title"]}. {publication} Available from: {location}')
references += [
    "10. Qwen Team. Qwen3-4B-Instruct-2507 [model card]. Hugging Face; 2025 [cited 2026 Nov 15]. Available from: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554",
    "11. Microsoft. Phi-3.5-mini-instruct [model card]. Hugging Face; 2024 [cited 2026 Nov 15]. Available from: https://huggingface.co/microsoft/Phi-3.5-mini-instruct/tree/2fe192450127e6a83f7441aef6e3ca586c338b77",
]

doc = document(paper["title"])
paragraph(doc, paper["title"].replace(" in two language models", "\nin two language models"), "Title")
paragraph(doc, author["display_name"])
paragraph(doc, author["affiliations"][0]["display"])
paragraph(doc, "Corresponding author: " + author["display_name"])
paragraph(doc, "Email: " + author["email"])
paragraph(doc, "ORCID: " + author["ORCID"])
paragraph(doc, "Short title: " + paper["short_title"])
doc.add_page_break()
paragraph(doc, "Abstract", "Heading 1")
paragraph(doc, paper["abstract"])
for block in paper["blocks"]:
    kind = block["kind"]
    if kind == "table":
        widths = {"behavior":[3.2,1.75,1.75], "stages":[1.35,1.7,1.9,1.75], "decomposition":[2.75,1.1,1.3,1.55]}[block["id"]]
        table(doc, block["title"], block["headers"], block["rows"], block["note"], widths)
    elif kind == "references":
        for entry in references:
            paragraph(doc, entry)
    else:
        paragraph(doc, block["text"], {"h1":"Heading 1", "h2":"Heading 2", "caption":"Caption"}.get(kind))
doc.save(OUT / "Manuscript.docx")

doc = document(supplement["title"])
paragraph(doc, "S1 Appendix", "Title")
paragraph(doc, supplement["title"])
for section in supplement["sections"]:
    heading = re.sub(r"^S(\d+)\.\s*", r"S\1 ", section["heading"])
    heading = re.sub(r"[^\w\s]", "", heading)
    paragraph(doc, heading, "Heading 1")
    for text in section["paragraphs"]:
        paragraph(doc, text)
    if "table" in section:
        values = section["table"]
        width = {2:[2.1,4.6], 3:[2.7,2.7,1.3], 5:[0.65,1.7,0.7,1.85,1.8]}[len(values["headers"])]
        table(doc, "", values["headers"], values["rows"], widths=width)
doc.save(OUT / "S1_Appendix.docx")

doc = document("Cover letter for PLOS ONE", double=False, numbers=False)
paragraph(doc, "Cover letter for PLOS ONE", "Title")
paragraph(doc, "16 November 2026")
paragraph(doc, "Dear Editors,")
paragraph(doc, 'Please consider the Research Article entitled "' + paper["title"] + '" for publication in PLOS ONE.')
paragraph(doc, "The study tests whether equivalent names retrieve consistent facts in two instruction-tuned language models. It evaluates 32 counterbalanced fictional registries per model and then tests a fixed internal-component repair hypothesis in one model. Wrong-object answers replicate across both model families, whereas the proposed repair fails its development criteria despite measurable improvements in prediction scores.")
paragraph(doc, "The contribution is a bounded empirical study of reference consistency and an audited negative intervention result. We distinguish it from prior library-alias, contextual-binding, and activation-patching work, and do not claim a new entity mechanism. The manuscript reports conditional sampling, repeated-world observations, the reused diagnostic cohort, negative intervention results, and the limits of inference.")
paragraph(doc, "Code and data package: " + release["repository"] + ", version " + release["version"] + ". " + release["native_scope"] + " Prior PLOS correspondence: [confirm details or none]. Suggested Academic Editors: [confirm suitable names]. Opposed reviewers: [confirm names and reasons or none].")
paragraph(doc, "[Before signing, confirm originality, exclusive submission, author approval, and any overlap with related work, including the article supplied for the author details.]")
paragraph(doc, "The author declares no competing interests.")
paragraph(doc, "Sincerely,\n" + author["display_name"] + "\n" + author["affiliations"][0]["display"] + "\n" + author["email"])
doc.save(OUT / "Cover_letter.docx")

doc = document("Submission checks and remaining author actions", double=False, numbers=False)
paragraph(doc, "Submission checks and remaining author actions", "Title")
paragraph(doc, "PLOS ONE requirements checked on 15 November 2026. This is a preparation package, not an uploaded submission. The negative findings are reported as findings, not as a newly established binding mechanism.")
paragraph(doc, "Author and declarations", "Heading 1")
paragraph(doc, author["display_name"] + ". " + author["affiliations"][0]["display"] + ". Email: " + author["email"] + ". ORCID: " + author["ORCID"])
paragraph(doc, "These details come from the exact article supplied by the author, cross-checked through its PubMed and DOI records. Confirm the contact role and current affiliation; authenticate the ORCID personally in Editorial Manager. Article source: https://pubmed.ncbi.nlm.nih.gov/42860077/ and https://api.crossref.org/works/10.1016/j.compbiolchem.2026.109438")
paragraph(doc, "Confirmed funding statement for the online Financial Disclosure field only: The author received no specific funding for this work. Do not paste funding into the manuscript or Acknowledgments.")
paragraph(doc, "Confirmed competing interests statement: The author declares no competing interests.")
paragraph(doc, "Still required: confirm CRediT roles, acknowledgments, final approval, originality/exclusive submission, related manuscripts, exact AI tool/model details, and actual human review of the code, evidence, references and claims. The contact record alone does not establish any of these confirmations.")
paragraph(doc, "Public data and code", "Heading 1")
paragraph(doc, availability_statement())
paragraph(doc, "Availability review still required: confirm that the released data are sufficient for the manuscript's findings and journal policy. Do not describe a private cloud drive, hash manifest or Git tag as a permanent public archive. If more native-score evidence is needed, arrange a suitable public deposit or agree an appropriate reproducibility minimum with the journal.")
paragraph(doc, "Scientific review before submission", "Heading 1")
paragraph(doc, "Confirm the intended original contribution relative to the closest studies. The models share 32 worlds; repeated questions are not independent worlds. Causal intervention results come from Qwen alone. The final six-world diagnosis reuses development data. The 64 reserved worlds remain unevaluated. Practical criteria are not significance tests. Canonical-token score changes are not repaired answers. Phi identifier formatting is not evidence of absent identity knowledge.")
paragraph(doc, "Verify all author-reviewed claims and exact citations, including the unusual DOI strings deposited by two 2026 ACL records. The manuscript's AI disclosure is substantive because assistance covered research and code, not only language editing. Automated audits must not be described as independent human peer review.")
paragraph(doc, "Files and format", "Heading 1")
for requirement in rules["submission_fields"]["requirements"]:
    paragraph(doc, requirement)
paragraph(doc, "For LaTeX submission, upload Manuscript.pdf; figures are separate. Its editable sources use the official PLOS template and bibliography style. Manuscript.docx is the alternative editable Word version, not an additional main submission file.")
paragraph(doc, "Separate figure files are supplied as RGB, flattened, LZW-compressed TIFFs at 600 dpi. Captions remain in the manuscript after first citation. Tables are editable. S1 Appendix and S1 Data have matching supporting captions. Review the rendered draft, resolve all bracketed author actions, and recheck current journal policy at the actual submission date.")
paragraph(doc, "Editorial portal actions", "Heading 1")
paragraph(doc, "Complete the one-page cover letter fields, choose appropriate editors and any opposed reviewers, and confirm any related manuscripts. Check the current publication charge and institutional support before submission; no fee has been paid. Building this package does not submit to the journal or contact an editor.")
paragraph(doc, "Official policy sources", "Heading 1")
p = doc.add_paragraph()
for label, url in (("Submission and formatting", rules["sources"]["submission"]), ("Publication criteria", rules["sources"]["publication_criteria"]), ("Data availability", rules["sources"]["data"]), ("Code sharing", rules["sources"]["code"]), ("AI disclosure", rules["sources"]["AI"]), ("Figures", rules["sources"]["figures"]), ("Supporting information", rules["sources"]["supporting"])):
    # Compact linked labels avoid an orphan page containing one long URL.
    if p._p.findall(qn("w:hyperlink")):
        p.add_run("; ")
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), doc.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True))
    run = OxmlElement("w:r")
    text_node = OxmlElement("w:t")
    text_node.text = label
    run.append(text_node)
    link.append(run)
    p._p.append(link)
    p.paragraph_format.line_spacing = 1
    p.paragraph_format.space_after = Pt(3)
    for run in p.runs:
        run.font.size = Pt(10.5)
doc.save(OUT / "Submission_checklist.docx")

# Figures are deterministic data plots drawn at publication resolution. No
# image generation or illustrative inference is involved.
DPI = 600
def font(size=9):
    # Resolve normal font-family names, never a machine-specific personal path.
    for family in ("Arial.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(family, round(size * DPI / 72))
        except OSError:
            continue
    raise RuntimeError("Install Arial, Liberation Sans or DejaVu Sans for figure export")

def text(draw, xy, value, size=9, fill="black", anchor="lt"):
    draw.text(xy, str(value), fill=fill, font=font(size), anchor=anchor)

def axes(draw, box, low, high, ticks):
    left, top, right, bottom = box
    draw.line((left,top,left,bottom,right,bottom), fill="black", width=4)
    for tick in ticks:
        y = bottom - (tick-low)/(high-low)*(bottom-top)
        draw.line((left,y,right,y), fill="#D7D7D7", width=2)
        text(draw,(left-35,y),tick,8,anchor="rm")
    return lambda value: bottom-(value-low)/(high-low)*(bottom-top)

world_data = {key:evidence["behavior"][key]["worlds"] for key in ("qwen","phi")}
assert len(world_data["qwen"]) == len(world_data["phi"]) == 32
assert sum(w["fact_correct"] for w in world_data["qwen"]) == 1657
assert sum(w["fact_correct"] for w in world_data["phi"]) == 1980
fig = Image.new("RGB", (4500,2700), "white")
draw = ImageDraw.Draw(fig)
colors = ("#0072B2", "#D55E00")
for panel, metric, title in ((0,"fact_accuracy","Correct factual answers (%)"),(1,"eligible_listed_wrong_object_discordant_pairs","Eligible equivalent name pairs (%)")):
    offset = panel*2200
    text(draw,(offset+100,55),"A" if panel==0 else "B",12)
    text(draw,(offset+380,85),title,10)
    ymax = 100 if panel==0 else 60
    box=(offset+270,250,offset+2020,2190)
    y = axes(draw,box,0,ymax,list(range(0,ymax+1,20)))
    x_positions=(offset+700,offset+1610)
    for index,(first,second) in enumerate(zip(world_data["qwen"],world_data["phi"])):
        assert first["world_id"] == second["world_id"]
        jitter=((index*17)%31-15)*8
        values=[first[metric]*100,second[metric]*100] if panel==0 else [first[metric]/24*100,second[metric]/24*100]
        draw.line((x_positions[0]+jitter,y(values[0]),x_positions[1]+jitter,y(values[1])),fill="#C5C5C5",width=2)
        for group,value in enumerate(values):
            cx,cy=x_positions[group]+jitter,y(value)
            draw.ellipse((cx-15,cy-15,cx+15,cy+15),fill=colors[group],outline="white",width=2)
    for xpos,label in zip(x_positions,("Qwen3 4B","Phi 3.5 mini")):
        text(draw,(xpos,2280),label,10,anchor="mt")
    text(draw,(offset+1120,2480),"32 paired worlds",9,anchor="mt")
fig.save(OUT/"Fig1.tif",compression="tiff_lzw",dpi=(DPI,DPI))
fig.save(FIGURES/"Fig1.png",dpi=(DPI,DPI))
fig.resize((1500,900)).save(QA/"Fig1.png")

effects=sealed(DATA/"cloud_reference_decomposition_metadata_20261115/reference-decomposition-20261115-effect-readout.json")
result=sealed(DATA/"cloud_reference_decomposition_metadata_20261115/decomposition/RESULT.json")
order=[("self","Self control"),("oldDcolorMass","Original color profile"),("sameFieldMass","Same field profile"),("sameFieldWithinOnly","Within record pattern"),("sameFieldProfileWithin","Profile and pattern"),("sameFieldRecordContribution","Record contribution"),("sameFieldNonrecordResidual","Nonrecord contribution"),("sameFieldFullHead","Whole donor head"),("normMatchedRandomFullHead","Matched random")]
points=[row for row in effects["per_question"] if row["fixed_old_W_token_id"] is not None]
worlds=sorted({row["world_id"] for row in points})
palette=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#666666"]
low=min(row["canonical_E_minus_W_change"] for row in points)
high=max(row["canonical_E_minus_W_change"] for row in points)
low,high=min(-2,math.floor(low/2)*2),max(8,math.ceil(high/2)*2)
fig=Image.new("RGB",(4500,3400),"white")
draw=ImageDraw.Draw(fig)
text(draw,(1150,50),"A",12)
text(draw,(3550,50),"B",12)
text(draw,(2250,105),"Correct minus original wrong token score change",10,anchor="mt")
text(draw,(3990,105),"Errors repaired",10,anchor="mt")
left,right,top,bottom=1260,3480,400,2730
x=lambda value:left+(value-low)/(high-low)*(right-left)
for tick in range(int(low),int(high)+1,2):
    draw.line((x(tick),top-70,x(tick),bottom+80),fill="#B5B5B5" if tick==0 else "#E1E1E1",width=4 if tick==0 else 2)
    text(draw,(x(tick),bottom+145),tick,8,anchor="mt")
for index,(key,label) in enumerate(order):
    cy=top+index*(bottom-top)/8
    text(draw,(1150,cy),label,8.5,anchor="rm")
    observations=[row for row in points if row["condition"]==key]
    assert len(observations)==8
    for obs_index,row in enumerate(observations):
        px,py=x(row["canonical_E_minus_W_change"]),cy+((obs_index%3)-1)*30
        draw.ellipse((px-13,py-13,px+13,py+13),fill=palette[worlds.index(row["world_id"])])
    mean=effects["by_condition"][key]["mean_canonical_E_minus_W_change_on_old_errors"]
    draw.line((x(mean)-22,cy-35,x(mean)+22,cy+35),fill="black",width=7)
    draw.line((x(mean)-22,cy+35,x(mean)+22,cy-35),fill="black",width=7)
    cell=result["assessment"]["conditions"][key]
    text(draw,(3990,cy),f'{cell["repaired"]}/8',10,anchor="mm")
text(draw,((left+right)//2,bottom+280),"Logit margin change",9,anchor="mt")
for index,world in enumerate(worlds):
    px=250+index*690
    draw.ellipse((px,3130,px+25,3155),fill=palette[index])
    text(draw,(px+45,3120),"World "+world.split(":")[-1],8)
fig.save(OUT/"Fig2.tif",compression="tiff_lzw",dpi=(DPI,DPI))
fig.save(FIGURES/"Fig2.png",dpi=(DPI,DPI))
fig.resize((1500,1133)).save(QA/"Fig2.png")

# Repackage the released processed data and historical executable sources.
# This small supplement intentionally excludes large selected captures and the
# unavailable full raw native-logit archive; it makes no complete-replay claim.
with ZipFile(OUT/"S1_Data.zip","w",ZIP_DEFLATED) as bundle:
    def put(name,value):
        bundle.writestr(name,json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    for path in sorted(PROCESSED.rglob("*")):
        if path.is_file():
            bundle.write(path, str(path.relative_to(PROCESSED)))
    for path in sorted((ROOT/"experiments").rglob("*.py")):
        bundle.write(path,"source/"+str(path.relative_to(ROOT/"experiments")))
    for name in ("descriptive_world_bootstrap.py","test_descriptive_world_bootstrap.py"):
        bundle.write(HERE/name,"paper_analysis/"+name)

qa_report={"abstract_words":len(paper["abstract"].split()),"body_words":sum(len(b.get("text","").split()) for b in paper["blocks"]),"source_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.json")},"files":{p.name:p.stat().st_size for p in OUT.iterdir()}}
assert all((OUT/name).stat().st_size<10_000_000 for name in ("Fig1.tif","Fig2.tif"))
assert (OUT/"S1_Data.zip").stat().st_size<20_000_000
(QA/"build_report.json").write_text(json.dumps(qa_report,indent=2)+"\n")
print(json.dumps(qa_report,indent=2))
