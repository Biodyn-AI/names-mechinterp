"""Fixed public geography facts: model-free schema and convention checks."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from e30_14_protocol import build_entity_panel

FACTS_PATH = Path(__file__).with_name("us_state_facts_2020.json")


def read_facts():
    return json.loads(FACTS_PATH.read_text(encoding="utf-8"))


def test_exactly_the_fifty_protocol_states_and_two_relations():
    fixed = read_facts()
    assert fixed["population_reference"] == "2020_census"
    assert set(fixed["facts"]) == {row["entity_id"] for row in build_entity_panel()}
    assert len(fixed["facts"]) == 50
    for relations in fixed["facts"].values():
        assert set(relations) == {"capital", "largest_city"}
        assert all(
            isinstance(city, str) and city.strip() == city and city
            for city in relations.values()
        )


def test_aliases_and_identity_flags_are_complete_and_unambiguous():
    fixed = read_facts()
    assert set(fixed["answer_aliases"]) == set(fixed["facts"])
    assert set(fixed["capital_largest_city_distinct"]) == set(fixed["facts"])
    for state, relations in fixed["facts"].items():
        aliases = fixed["answer_aliases"][state]
        assert set(aliases) == set(relations)
        for relation, canonical in relations.items():
            assert canonical in aliases[relation]
            assert aliases[relation] and len(aliases[relation]) == len(
                set(aliases[relation])
            )
        capital = {alias.casefold() for alias in aliases["capital"]}
        largest = {alias.casefold() for alias in aliases["largest_city"]}
        assert fixed["capital_largest_city_distinct"][state] == capital.isdisjoint(
            largest
        )
    assert (
        sum(
            not distinct for distinct in fixed["capital_largest_city_distinct"].values()
        )
        == 17
    )


def test_census_2020_edge_cases_and_name_aliases():
    fixed = read_facts()
    assert fixed["facts"]["alabama"] == {
        "capital": "Montgomery",
        "largest_city": "Huntsville",
    }
    assert fixed["facts"]["tennessee"] == {
        "capital": "Nashville",
        "largest_city": "Nashville",
    }
    assert fixed["facts"]["hawaii"] == {
        "capital": "Honolulu",
        "largest_city": "Honolulu",
    }
    assert not fixed["capital_largest_city_distinct"]["hawaii"]
    assert "Urban Honolulu CDP" in fixed["answer_aliases"]["hawaii"]["largest_city"]
    assert (
        "Nashville-Davidson metropolitan government (balance)"
        in fixed["answer_aliases"]["tennessee"]["largest_city"]
    )
    assert fixed["facts"]["kentucky"]["largest_city"] == "Louisville"
    assert fixed["facts"]["indiana"]["largest_city"] == "Indianapolis"
    assert "St. Paul" in fixed["answer_aliases"]["minnesota"]["capital"]
    assert "New York" in fixed["answer_aliases"]["new_york"]["largest_city"]


def test_primary_provenance_and_semantic_payload_checksum():
    fixed = read_facts()
    sources = fixed["provenance"]["sources"]
    for source in sources.values():
        host = urlparse(source["url"]).hostname
        assert host and host.endswith((".census.gov", ".usgs.gov"))
        if "sha256" in source:
            assert re.fullmatch("[0-9a-f]{64}", source["sha256"])
    assert (
        fixed["provenance"]["census_api_attempt"]["status"]
        == "unavailable_without_api_key"
    )
    assert (
        fixed["provenance"]["largest_city_derivation"]
        == "Transcribed the official Census published 2020 largest-place table; not recomputed from the unavailable API."
    )
    assert fixed["uncertainties"] == []
    payload = {
        key: fixed[key]
        for key in (
            "facts",
            "answer_aliases",
            "capital_largest_city_distinct",
            "population_reference",
        )
    }
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    assert digest == fixed["semantic_payload_sha256"]
