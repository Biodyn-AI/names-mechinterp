"""One-shot orchestration, native-logit custody, and a simulated full run."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import e30_14_capture as capture
import e30_14_protocol as protocol
import e30_14_run as run
from test_e30_14_capture import CharacterTokenizer


def test_normalization_does_not_turn_extra_words_into_a_city():
    assert run.normalize_city("  NEW   YORK CITY. \n") == "new york city"
    assert run.normalize_city("The city is Austin.") != run.normalize_city("Austin")
    assert run.normalize_city("Austin, Texas") != run.normalize_city("Austin")


def test_shared_public_blob_layout_is_allowed_but_real_escape_is_rejected(tmp_path):
    cache_root = tmp_path / "public-cache"
    repository = cache_root / "models--public--model"
    snapshot = repository / "snapshots" / "fixed-revision"
    snapshot.mkdir(parents=True)
    shared_key = "ab" + "1" * 62
    shared_blob = cache_root / "blobs" / "ab" / shared_key
    shared_blob.parent.mkdir(parents=True)
    shared_blob.write_text("synthetic public tokenizer")
    repository_blob = repository / "blobs" / ("2" * 64)
    repository_blob.parent.mkdir(parents=True)
    repository_blob.symlink_to(shared_blob)
    asset = snapshot / "tokenizer.json"
    asset.symlink_to(repository_blob)
    assert run.snapshot_file(asset, snapshot) == shared_blob
    external_file = tmp_path / "outside-public-cache.txt"
    external_file.write_text("synthetic outside file, not a real credential")
    escaped_asset = snapshot / "escaped.json"
    escaped_asset.symlink_to(external_file)
    with pytest.raises(ValueError, match="escaped"):
        run.snapshot_file(escaped_asset, snapshot)


@pytest.mark.parametrize(
    "relative", ["unrelated.json", "blobs/ab/not-a-key", "blobs/ff/" + "a" * 64]
)
def test_shared_blob_support_does_not_admit_arbitrary_cache_files(tmp_path, relative):
    cache_root = tmp_path / "public-cache"
    snapshot = cache_root / "models--public--model" / "snapshots" / "fixed-revision"
    snapshot.mkdir(parents=True)
    target = cache_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("synthetic non-asset")
    asset = snapshot / "tokenizer.json"
    asset.symlink_to(target)
    with pytest.raises(ValueError, match="escaped"):
        run.snapshot_file(asset, snapshot)


def test_shared_blob_root_symlink_cannot_escape_public_cache(tmp_path):
    cache_root = tmp_path / "public-cache"
    snapshot = cache_root / "models--public--model" / "snapshots" / "fixed-revision"
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside"
    target = outside / "ab" / ("ab" + "1" * 62)
    target.parent.mkdir(parents=True)
    target.write_text("synthetic outside blob")
    (cache_root / "blobs").symlink_to(outside)
    asset = snapshot / "tokenizer.json"
    asset.symlink_to(cache_root / "blobs" / "ab" / target.name)
    with pytest.raises(ValueError, match="escaped"):
        run.snapshot_file(asset, snapshot)


def test_native_shards_are_bounded_and_hash_verified(tmp_path):
    writer = run.RawWriter(tmp_path, shard_size=2)
    for index in range(5):
        writer.add(f"endpoint_{index}", torch.arange(3, dtype=torch.float32) + index)
    index = writer.finish()
    assert len(index["shards"]) == 3
    assert [shard["endpoint_count"] for shard in index["shards"]] == [2, 2, 1]
    reader = run.RawReader(tmp_path, index)
    assert torch.equal(reader.read("endpoint_4"), torch.tensor([4.0, 5.0, 6.0]))
    with pytest.raises(ValueError, match="float32"):
        writer.add("bad", torch.ones(3, dtype=torch.float64))
    shard = tmp_path / index["shards"][0]["file"]
    shard.write_bytes(shard.read_bytes() + b"damage")
    with pytest.raises(ValueError, match="checksum"):
        run.RawReader(tmp_path, index).read("endpoint_0")


def test_snapshot_files_cannot_read_outside_the_public_blob_store(tmp_path):
    snapshot = tmp_path / "repository" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (tmp_path / "repository" / "blobs").mkdir()
    blob = tmp_path / "repository" / "blobs" / "public_hash"
    blob.write_text("synthetic public configuration")
    (snapshot / "config.json").symlink_to(blob)
    assert run.snapshot_file(snapshot / "config.json", snapshot) == blob
    outside = tmp_path / "outside"
    outside.write_text("not part of the snapshot")
    (snapshot / "tokenizer.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escaped"):
        run.snapshot_file(snapshot / "tokenizer.json", snapshot)


def test_real_qualification_loop_uses_complete_names_and_eight_alias_matches(
    monkeypatch,
):
    facts = json.loads(Path(__file__).with_name("us_state_facts_2020.json").read_text())
    rows = protocol.build_qualification_prompts(facts)
    for index, row in enumerate(rows):
        row["input_ids"] = [800 + index]
    entities = sorted({row["entity_id"] for row in rows})

    class Tokenizer:
        eos_token_id = 0

        def decode(self, ids, **kwargs):
            return "".join(chr(token - 1) for token in ids if token)

    class ScriptedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.config = SimpleNamespace(eos_token_id=0)

        def forward(self, input_ids, **kwargs):
            assert kwargs["logits_to_keep"] == 1 and kwargs["use_cache"] is False
            row = rows[int(input_ids[0, 0]) - 800]
            answer = facts["facts"][row["entity_id"]][row["relation_id"]]
            step = input_ids.shape[1] - 1
            logits = torch.full((1, 1, 1000), -20.0, dtype=torch.float32)
            token = ord(answer[step]) + 1 if step < len(answer) else 0
            logits[0, 0, token] = 10.0
            logits[0, 0, 500 + entities.index(row["entity_id"])] = 5.0
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(run, "check_deadline", lambda deadline: None)
    report, logits = run.qualify_model(
        ScriptedModel().eval(), Tokenizer(), rows, facts, deadline=run.DEADLINE
    )
    assert report["passed"] and report["correct"] == 32 and len(logits) == 32
    assert len(report["nearest_output"]["capital"]) == 8
    assert all(item["correct"] for item in report["nearest_output"]["capital"])
    assert all(len(item["generated_ids"]) <= 12 for item in report["rows"])
    assert report["physical_forward_count"] == sum(
        len(item["generated_ids"]) for item in report["rows"]
    )


@pytest.fixture
def simulation(tmp_path, monkeypatch):
    real = run.modules()
    design = protocol.build_protocol()
    design.update(
        rank_grid=[2],
        ridge_fraction_grid=[0.1],
        shuffle_count=2,
        shuffles=design["shuffles"][:2],
    )
    fake_protocol = SimpleNamespace(
        build_protocol=lambda: deepcopy(design),
        build_prompts=protocol.build_prompts,
        build_qualification_prompts=protocol.build_qualification_prompts,
    )
    source = tmp_path / "source"
    source.mkdir()
    for name in run.REQUIRED_SOURCES:
        (source / name).write_text("# synthetic frozen source\n")
    facts_path = Path(__file__).with_name("us_state_facts_2020.json")
    (source / facts_path.name).write_bytes(facts_path.read_bytes())
    events, state = [], {"qualified": True, "model_name": "modern"}
    output = tmp_path / "output"

    class Executor:
        def __init__(self, model, blocks, **kwargs):
            self.physical_forward_count = 0

        def capture_prompts(self, rows):
            is_test = rows[0]["split"] == "test"
            if is_test:
                assert (output / state["model_name"] / "SELECTION.json").exists()
                assert len(rows) == 160
            else:
                assert len(rows) == 64 and {row["split"] for row in rows} == {
                    "fit",
                    "validation",
                }
            events.append("test_capture" if is_test else "fit_validation_capture")
            values, logits = {}, {}
            entity_ids = [entry["entity_id"] for entry in protocol.build_entity_panel()]
            for row in rows:
                entity = entity_ids.index(row["entity_id"]) + 1
                vector = torch.tensor(
                    [entity, entity**2 / 50, entity % 7, 1.0], dtype=torch.float32
                )
                if row["form"] == "abbr":
                    vector = vector.flip(0)
                values[row["prompt_id"]] = {
                    f"residual::{layer}::{site}": vector.clone()
                    for layer in range(2)
                    for site in ("marker", "query")
                }
                logits[row["prompt_id"]] = (
                    torch.arange(600, dtype=torch.float32).remainder(entity + 2) / 100
                )
            self.clean = logits
            self.physical_forward_count += len(rows)
            return {
                "states": values,
                "clean_logits": logits,
                "physical_forward_count": len(rows),
            }

        def qualify_oracles(self, **kwargs):
            events.append("oracles")
            assert (output / state["model_name"] / "SELECTION.json").exists()
            self.physical_forward_count += 1
            return {"passed": True, "physical_forward_count": 1}

        def run_patch(self, prompt_id, **kwargs):
            assert (output / state["model_name"] / "PLAN.pt").exists()
            assert (output / state["model_name"] / "PLAN.json").exists()
            self.physical_forward_count += 1
            operand = kwargs.get("delta", kwargs.get("replacement"))
            return self.clean[prompt_id] + operand.float().sum() * torch.linspace(
                0, 0.001, 600
            )

        def close(self):
            events.append("close")

    def capture_stage(model, rows, **kwargs):
        executor = Executor(model, [])
        return executor, executor.capture_prompts(rows)

    fake_capture = SimpleNamespace(
        **{
            name: getattr(capture, name)
            for name in (
                "prepare_token_rows",
                "make_token_manifest",
                "estimate_storage",
                "select_maps",
                "selection_metadata",
            )
        },
        capture_stage=capture_stage,
        model_blocks=lambda *args: [],
    )
    fake_alignment = SimpleNamespace(ResidualPatchExecutor=Executor)
    monkeypatch.setattr(
        run,
        "modules",
        lambda: SimpleNamespace(
            protocol=fake_protocol,
            capture=fake_capture,
            alignment=fake_alignment,
            evaluate=real.evaluate,
            summary=real.summary,
        ),
    )

    def assets(info, **kwargs):
        assert (output / "DESIGN.json").exists() and (output / "SOURCES.json").exists()
        events.append("tokenizer")
        return {
            "snapshot": str(source),
            "tokenizer": CharacterTokenizer(),
            "config": {"vocab_size": 600, "hidden_size": 4, "num_hidden_layers": 2},
            "asset_sha256": {},
        }

    def load_model(*args):
        assert (output / "REGISTRATION.json").exists()
        events.append("weights")
        run.seal_json(
            args[2],
            {
                "model_id": args[0]["model_id"],
                "revision": args[0]["revision"],
                "files": {},
            },
        )
        return object()

    def qualification(model, tokenizer, rows, facts, **kwargs):
        events.append("qualification")
        return {
            "passed": state["qualified"],
            "physical_forward_count": 32,
            "rows": [
                {
                    "prompt_id": row["prompt_id"],
                    "generated_ids": [1],
                    "generated_text": "synthetic",
                }
                for row in rows
            ],
        }, {row["prompt_id"]: torch.zeros(600, dtype=torch.float32) for row in rows}

    monkeypatch.setattr(run, "load_token_assets", assets)
    monkeypatch.setattr(run, "load_model", load_model)
    monkeypatch.setattr(run, "load_tokenizer", lambda snapshot: CharacterTokenizer())
    monkeypatch.setattr(
        run, "configure_runtime", lambda registration: {"synthetic": True}
    )
    monkeypatch.setattr(run, "qualify_model", qualification)
    monkeypatch.setattr(run, "check_deadline", lambda deadline: None)
    return output, source, events, state


def test_full_simulated_run_seals_selection_and_plan_before_test_patches(simulation):
    output, source, events, _ = simulation
    run.prepare(output, source_root=source)
    assert events == ["tokenizer", "tokenizer"]
    result = run.run_model(output, "modern", source_root=source)
    assert result["status"] == "completed"
    assert events.index("weights") > events.index("tokenizer")
    assert (
        events.index("qualification")
        < events.index("fit_validation_capture")
        < events.index("test_capture")
        < events.index("oracles")
    )
    model_root = output / "modern"
    assert (model_root / "RAW_INDEX.json").exists()
    assert (model_root / "CELL_SCORES.json").exists()
    diagnostics = run.read_json(model_root / "HELDOUT_CLEAN_FACT_DIAGNOSTICS.json")
    assert diagnostics["first_token_only_not_whole_city_accuracy"] is True
    assert diagnostics["global"]["count"] == 160
    assert run.rescore(output, "modern", source_root=source)["exact_raw_rescore_match"]
    custody = run.read_json(model_root / "CUSTODY.json")
    assert custody["artifact_sha256"]["PLAN.pt"] == run.sha256(model_root / "PLAN.pt")
    assert custody["artifact_sha256"]["RAW_INDEX.json"] == run.sha256(
        model_root / "RAW_INDEX.json"
    )
    with pytest.raises(FileExistsError):
        run.run_model(output, "modern", source_root=source)


def test_semantic_failure_prevents_every_fresh_capture(simulation):
    output, source, events, state = simulation
    run.prepare(output, source_root=source)
    state["qualified"] = False
    assert (
        run.run_model(output, "modern", source_root=source)["status"]
        == "semantic_qualification_failed"
    )
    assert "fit_validation_capture" not in events and "test_capture" not in events
    assert (output / "modern" / "QUALIFICATION.pt").exists()
    assert not (output / "modern" / "SELECTION.json").exists()


def test_medium_bridge_proceeds_even_when_semantic_qualification_fails(simulation):
    output, source, events, state = simulation
    run.prepare(output, source_root=source)
    state.update(qualified=False, model_name="medium")
    result = run.run_model(output, "medium", source_root=source)
    assert result["status"] == "completed"
    assert (
        result["interpretation"]
        == "computational bridge, not behavior-qualified semantics"
    )
    assert "fit_validation_capture" in events and "test_capture" in events
    assert not run.read_json(output / "medium" / "QUALIFICATION.json")["passed"]


def test_source_change_and_prepare_clobber_are_rejected_before_weights(simulation):
    output, source, events, _ = simulation
    run.prepare(output, source_root=source)
    with pytest.raises(FileExistsError):
        run.prepare(output, source_root=source)
    (source / run.REQUIRED_SOURCES[0]).write_text("# changed\n")
    with pytest.raises(ValueError, match="source"):
        run.run_model(output, "modern", source_root=source)
    assert "weights" not in events
