"""Model-free and synthetic safeguards for the new, separately frozen study."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

try:
    import torch
except ImportError:
    torch = None

import confirmation_run as runner


class Tokenizer:
    eos_token_id = 3

    def decode(self, ids, *, skip_special_tokens):
        assert not skip_special_tokens
        return "".join({0: "red", 1: "blue", 2: "<user>", 3: "<eos>"}[i] for i in ids)


class Model:
    config = SimpleNamespace(eos_token_id=3)
    generation_config = SimpleNamespace(eos_token_id=[3])

    def __init__(self, *, endless=False):
        self.parameter = torch.zeros(1)
        self.calls = []
        self.endless = endless

    def parameters(self):
        yield self.parameter

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        scores = (
            [4.0, 1.0, 0.0, -1.0]
            if self.endless or len(self.calls) == 1
            else [0.0, 1.0, 2.0, 4.0]
        )
        return SimpleNamespace(logits=torch.tensor([[scores]]))


class MemoryRaw:
    def __init__(self):
        self.saved = {}
        self.index = {"endpoints": self.saved}

    def add(self, identity, scores):
        assert identity not in self.saved
        self.saved[identity] = scores.clone()

    def read(self, identity):
        return self.saved[identity]


def behavior():
    def decode(tokenizer, ids, eos):
        return tokenizer.decode(
            ids[:-1] if ids and ids[-1] in eos else ids, skip_special_tokens=False
        )

    return SimpleNamespace(
        decode_answer=decode,
        eos_ids=lambda model, tokenizer: {3},
    )


def generate(model=None, *, cap=1000, deadline=None):
    model, raw = model or Model(), MemoryRaw()
    row = {"id": "confirmation:000:fact", "input_ids": [8, 9], "expected_answer": "red"}
    prediction = runner.generate(
        model,
        Tokenizer(),
        row,
        row["id"],
        raw,
        behavior=behavior(),
        vocab_size=4,
        deadline=deadline or (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        raw_cap=cap,
        counter={"bytes": 0},
    )
    return model, raw, row, prediction


@pytest.mark.skipif(torch is None, reason="optional research dependencies")
def test_generation_is_complete_native_greedy_and_never_cached():
    model, raw, row, prediction = generate()
    assert prediction["generated_ids"] == [0, 3]
    assert prediction["generated_text"] == "red"
    assert prediction["terminated"] and not prediction["truncated"]
    assert len(raw.saved) == 2
    assert [c["input_ids"].tolist() for c in model.calls] == [[[8, 9]], [[8, 9, 0]]]
    assert all(
        c["use_cache"] is False and c["logits_to_keep"] == 1 for c in model.calls
    )
    runner.audit_predictions(
        [row],
        [prediction],
        raw,
        Tokenizer(),
        {3},
        behavior().decode_answer,
        vocab_size=4,
    )


@pytest.mark.skipif(torch is None, reason="optional research dependencies")
def test_eight_steps_without_termination_remain_a_truncated_error():
    _, raw, row, prediction = generate(Model(endless=True))
    assert prediction["generated_ids"] == [0] * 8
    assert prediction["truncated"] and not prediction["terminated"]
    runner.audit_predictions(
        [row],
        [prediction],
        raw,
        Tokenizer(),
        {3},
        behavior().decode_answer,
        vocab_size=4,
    )


@pytest.mark.skipif(torch is None, reason="optional research dependencies")
def test_raw_limit_and_deadline_fail_before_any_forward_or_write():
    model = Model()
    with pytest.raises(RuntimeError, match="ceiling"):
        generate(model, cap=15)
    assert not model.calls
    model = Model()
    with pytest.raises(RuntimeError, match="deadline"):
        generate(model, deadline=(datetime.now(UTC) - timedelta(seconds=1)).isoformat())
    assert not model.calls


@pytest.mark.parametrize(
    "change", ["token", "decode", "termination", "missing", "extra", "expected"]
)
@pytest.mark.skipif(torch is None, reason="optional research dependencies")
def test_independent_native_audit_rejects_tampering(change):
    _, raw, row, prediction = generate()
    if change == "token":
        prediction["generated_ids"][0] = 1
    elif change == "decode":
        prediction["generated_text"] = "blue"
    elif change == "termination":
        prediction["terminated"] = False
    elif change == "missing":
        del raw.saved[prediction["step_endpoint_ids"][0]]
    elif change == "extra":
        raw.add("unregistered::step00", torch.zeros(4))
    else:
        prediction["expected_answer"] = "blue"
    with pytest.raises((ValueError, KeyError)):
        runner.audit_predictions(
            [row],
            [prediction],
            raw,
            Tokenizer(),
            {3},
            behavior().decode_answer,
            vocab_size=4,
        )


def test_token_reattachment_changes_only_machine_specific_paths():
    archived = {
        "rows": [{"id": "old", "input_ids": [1]}],
        "asset_sha256": {"a": "digest"},
        "vocab_size": 4,
        "eos_token_ids": [3],
        "snapshot": "/old",
        "cache_root": "/old-cache",
        "model_outputs_used": False,
    }
    current = {
        "asset_sha256": {"a": "digest"},
        "vocab_size": 4,
        "eos_token_ids": [3],
        "snapshot": "/new",
        "cache_root": "/new-cache",
    }
    attached = runner.reattach_tokens(archived, current)
    assert attached == {**archived, "snapshot": "/new", "cache_root": "/new-cache"}
    assert archived["snapshot"] == "/old"
    for key, changed in (
        ("asset_sha256", {"a": "changed"}),
        ("vocab_size", 5),
        ("eos_token_ids", [2]),
    ):
        with pytest.raises(ValueError, match="archived"):
            runner.reattach_tokens(archived, {**current, key: changed})


def test_confirmation_grid_cannot_be_shortened_or_substituted():
    rows = [
        {
            "id": f"r{i}",
            "world_id": f"confirmation:{i // 96:03d}",
            "split": "confirmation",
        }
        for i in range(6144)
    ]
    worlds = [
        {"world_id": f"confirmation:{i:03d}", "split": "confirmation"}
        for i in range(64)
    ]
    assert runner.confirmation_grid(rows, worlds) == rows
    with pytest.raises(ValueError, match="64-world"):
        runner.confirmation_grid(rows[:-1], worlds)
    with pytest.raises(ValueError, match="64-world"):
        runner.confirmation_grid(rows[:-1] + [rows[0]], worlds)
    with pytest.raises(ValueError, match="64-world"):
        runner.confirmation_grid(
            [{**rows[0], "split": "development"}, *rows[1:]], worlds
        )


def test_behavioral_native_records_cannot_refer_to_a_different_question():
    rows = [{"id": "first"}, {"id": "second"}]
    predictions = [
        {"id": "first", "row_id": "first"},
        {"id": "second", "row_id": "second"},
    ]
    runner.behavior_coverage(rows, predictions)
    with pytest.raises(ValueError, match="literal row"):
        runner.behavior_coverage(
            rows, [{**predictions[0], "row_id": "second"}, predictions[1]]
        )
    with pytest.raises(ValueError, match="literal row"):
        runner.behavior_coverage(rows, [predictions[0], predictions[0]])


def test_weight_loading_cannot_consume_registered_output_space(tmp_path, monkeypatch):
    calls, sealed, spaces = [], {}, iter((100, 99))
    old = SimpleNamespace(
        seal_json=lambda path, content: sealed.update({path.name: content})
    )
    frozen = {"model_limits_bytes": {"qwen": 100}}
    behavior = SimpleNamespace(
        MODELS={"qwen": {}}, validate_deadline=lambda deadline: None
    )
    monkeypatch.setattr(
        runner, "check", lambda output: (None, behavior, old, {}, {}, frozen)
    )
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=next(spaces))
    )
    monkeypatch.setattr(
        runner, "load_runtime", lambda *args: calls.append("load") or (None, None, None)
    )
    monkeypatch.setattr(
        runner,
        "native_forward",
        lambda *args: pytest.fail("model forward must remain forbidden"),
    )
    with pytest.raises(RuntimeError, match="after model loading"):
        runner.run_behavior(tmp_path, "qwen", deadline="unused synthetic deadline")
    assert calls == ["load"]
    assert "failure.json" in sealed and (tmp_path / "qwen").is_dir()


@pytest.mark.skipif(torch is None, reason="optional research dependencies")
def test_compact_capture_keeps_replay_inputs_without_all_head_rows():
    capture = {
        "installed_source_sha256": {"source": "digest"},
        "intervention": {"delta": torch.ones(2)},
        "layers": {
            i: {
                "attention_rows": torch.ones(2, 4),
                "record_value_sha256": str(i),
                "head_dim": 2,
                "record_positions": [0, 1],
            }
            for i in range(3)
        },
    }
    compact = runner.compact_capture(capture, 1)
    assert "attention_rows" not in compact["layers"][0]
    assert torch.equal(
        compact["layers"][1]["attention_rows"], capture["layers"][1]["attention_rows"]
    )
    assert compact["layers"][2]["record_value_sha256"] == "2"
    assert compact["intervention"] is capture["intervention"]


@pytest.mark.skipif(torch is None, reason="optional research dependencies")
def test_random_tiny_qwen_repair_and_independent_cpu_audit(tmp_path, monkeypatch):
    """Exercise orchestration only: no pretrained weights or study observations."""
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(65116)
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=12,
        max_position_embeddings=64,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=31,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen3ForCausalLM(config).float().eval().requires_grad_(False)

    class TinyTokenizer:
        eos_token_id = 31

        def decode(self, ids, *, skip_special_tokens):
            assert not skip_special_tokens
            return " ".join(str(i) for i in ids)

    tokenizer = TinyTokenizer()
    real_behavior = runner.module_at("error_run", runner.REPO / runner.SOURCE_DIRS[1])
    old = real_behavior.helpers(runner.REPO / runner.SOURCE_DIRS[-1])
    routing = runner.module_at("routing_run", runner.REPO / runner.SOURCE_DIRS[3])
    mass = runner.module_at("record_routing_hooks", runner.REPO / runner.SOURCE_DIRS[3])
    query = runner.module_at("query_delta_hooks", runner.REPO / runner.SOURCE_DIRS[2])
    regions = {obj: [index] for index, obj in enumerate("ABCD")}
    rows = [
        {
            "id": f"synthetic::{role}::{field}",
            "input_ids": [2, 3, 4, 5, 6 + role_index, 8 + field_index],
            "expected_answer": "synthetic only",
            "split": "confirmation",
        }
        for role_index, role in enumerate(("donor", "recipient"))
        for field_index, field in enumerate(runner.FIELDS)
    ]
    tokens = {
        "rows": rows,
        "vocab_size": 32,
        "eos_token_ids": [31],
        "snapshot": "not-used",
    }
    old.seal_json(tmp_path / "REGISTRATION.json", {"synthetic_test": True})
    baseline_root = tmp_path / "qwen"
    baseline_root.mkdir()
    writer = old.RawWriter(baseline_root)
    deadline = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    baseline = [
        runner.generate(
            model,
            tokenizer,
            row,
            row["id"],
            writer,
            behavior=real_behavior,
            vocab_size=32,
            deadline=deadline,
            raw_cap=100_000,
            counter={"bytes": 0},
        )
        for row in rows
    ]
    writer.finish()
    old.seal_json(baseline_root / "GENERATIONS.json", baseline)
    runner.seal_custody(
        old, baseline_root, old.verify_file(tmp_path / "REGISTRATION.json")
    )
    # These deliberately mocked qualification receipts test the execution code,
    # not real 64-world eligibility or any empirical result.
    old.seal_json(
        baseline_root / "NATIVE_AUDIT.json",
        {
            "exact_match": True,
            "required_rows": 6144,
            "custody_sha256": old.verify_file(baseline_root / "CUSTODY.json"),
        },
    )
    case = {
        "case_id": "synthetic",
        "world_id": "synthetic",
        "donor_row_ids": {
            field: f"synthetic::donor::{field}" for field in runner.FIELDS
        },
        "recipient_row_ids": {
            field: f"synthetic::recipient::{field}" for field in runner.FIELDS
        },
    }
    frozen = {
        "conditions": list(runner.CONDITIONS),
        "selected_head": {"layer_index": 0, "head_index": 0},
        "frozen_discovery_mean_Dcolor": [0.25] * 4,
        "numeric": {"max_query_head_drift": 2e-5},
        "model_limits_bytes": {"repair": 5_000_000_000},
    }
    plan = SimpleNamespace(
        select_cases=lambda *args: [case],
        assess_repair=lambda cases, rows, baseline, generations: {
            "synthetic_generation_count": len(generations)
        },
    )
    checked = (
        plan,
        real_behavior,
        old,
        {"repo_root": str(runner.REPO)},
        {"rows": rows, "worlds": []},
        frozen,
    )
    monkeypatch.setattr(runner, "check", lambda output: checked)
    monkeypatch.setattr(
        runner,
        "modules",
        lambda *args, **kwargs: (
            plan,
            real_behavior,
            old,
            routing,
            mass,
            query,
            SimpleNamespace(record_regions=lambda row: regions),
        ),
    )
    monkeypatch.setattr(
        runner, "load_runtime", lambda *args: (model, tokenizer, tokens)
    )
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=100_000_000_000)
    )
    monkeypatch.setattr(real_behavior, "checked_tokens", lambda *args: tokens)
    monkeypatch.setattr(old, "load_tokenizer", lambda path: tokenizer)
    runner.run_repair(tmp_path, deadline=deadline)
    audit = runner.audit_repair(tmp_path)
    assert audit["exact_match"] and audit["cases"] == 1
    assert audit["generations"] == 12
    assert old.read_json(tmp_path / "repair" / "RESULT.json") == {
        "synthetic_generation_count": 12
    }
    with pytest.raises(FileExistsError):
        runner.run_repair(tmp_path, deadline=deadline)
