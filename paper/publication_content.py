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
        introduction = "Code and study data are available in the public repository at "
    else:
        introduction = "The code and study-data package is prepared for release at "
    return (
        introduction + location + " (version " + release["version"]
        + "; release tag " + release.get("release_tag", release["version"]) + "). "
        "Code is licensed under " + release["code_license"] + "; study data, "
        "documentation and paper are licensed under " + release["data_paper_license"] + ". "
        "S1 Data contains the prompts, recorded answers, values plotted in the figures, "
        "and records showing where the data came from. "
        + release["native_scope"] + " "
        "The Git tag identifies this version but is not a permanent archive. "
        "No permanent archive identifier (DOI) is claimed."
    )


def load_manuscript():
    manuscript = json.loads((HERE / "manuscript.json").read_text())
    for index, block in enumerate(manuscript["blocks"]):
        if block.get("text") == "Data and code availability":
            manuscript["blocks"][index + 1]["text"] = availability_statement()
        if block.get("text", "").startswith("S1 Data."):
            block["text"] = (
                "S1 Data. Recorded answers and values used in the figures. Machine-readable prompts "
                "and model answers for 32 development worlds and the same 64 fresh confirmation worlds "
                "tested with both models, summaries by world, complete internal-change outcomes, score "
                "differences, source checksums, and plotted values. This small supporting archive does "
                "not contain large prediction-score or captured-tensor files. The repository and release "
                "assets state their separate availability and the checks that can be repeated."
            )
    return manuscript
