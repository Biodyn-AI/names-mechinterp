"""Release-scope wording shared by Word and LaTeX; never invent an archive DOI."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def release_metadata():
    return json.loads((HERE / "release_metadata.json").read_text())


def availability_statement():
    release = release_metadata()
    location = release["repository"]
    if release["public_release_verified"]:
        location += "/releases/tag/" + release["version"]
        introduction = "Code and study data are available in the public release at "
    else:
        introduction = "The code and study-data package is prepared for release at "
    return (
        introduction + location + " (" + release["version"] + "). "
        "Code is licensed under " + release["code_license"] + "; study data, "
        "documentation and paper are licensed under " + release["data_paper_license"] + ". "
        "S1 Data contains processed prompts, outcomes, figure coordinates and provenance records. "
        + release["native_scope"] + " "
        "A Git tag is a version identifier, not a permanent archive; no archive DOI is claimed."
    )


def load_manuscript():
    manuscript = json.loads((HERE / "manuscript.json").read_text())
    for index, block in enumerate(manuscript["blocks"]):
        if block.get("text") == "Data and code availability":
            manuscript["blocks"][index + 1]["text"] = availability_statement()
        if block.get("text", "").startswith("S1 Data. Processed outcomes"):
            block["text"] = (
                "S1 Data. Processed outcomes and figure source data. Machine-readable prompts "
                "and behavioral answers for the 32 evaluated worlds, per-world summaries, "
                "fixed-component outcomes and score changes, source hashes and figure coordinates. "
                "This processed supplement is distinct from the release's selected native artifacts; "
                "the release-scope statement specifies which historical audits can be repeated."
            )
    return manuscript
