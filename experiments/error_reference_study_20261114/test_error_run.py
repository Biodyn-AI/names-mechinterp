"""Safety contracts for the separate wrong-object behavioral runner."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import error_run as runner


def test_deadline_must_be_fresh_explicit_utc_and_bounded():
    now = datetime(2026, 11, 14, tzinfo=UTC)
    runner.validate_deadline("2026-11-14T01:00:00Z", now=now)
    for value in (
        None,
        "2026-11-14T01:00:00",
        "2026-11-14T00:00:00Z",
        "2026-11-14T07:00:00Z",
    ):
        with pytest.raises((ValueError, TypeError)):
            runner.validate_deadline(value, now=now)


def test_confirmation_is_forbidden_and_development_count_is_exact():
    rows = [{"id": str(i), "split": "development"} for i in range(3072)]
    selected = runner.development_rows(
        {"rows": rows + [{"id": "held", "split": "confirmation"}]}, "development"
    )
    assert len(selected) == 3072 and all(
        row["split"] == "development" for row in selected
    )
    with pytest.raises(ValueError, match="confirmation"):
        runner.development_rows({"rows": rows}, "confirmation")
    with pytest.raises(ValueError, match="exact"):
        runner.development_rows({"rows": rows[:-1]}, "development")
    with pytest.raises(ValueError, match="exact"):
        runner.development_rows({"rows": rows[:-1] + [rows[0]]}, "development")


def test_analysis_receives_only_the_32_development_worlds():
    worlds = [
        {"world_id": f"development:{i}", "split": "development"} for i in range(32)
    ]
    worlds += [{"world_id": "confirmation:0", "split": "confirmation"}]
    assert runner.development_worlds(worlds) == worlds[:32]
    with pytest.raises(ValueError, match="32"):
        runner.development_worlds(worlds[1:])


def test_storage_estimate_covers_all_steps_and_uses_output_vocabulary():
    assert (
        runner.estimated_bytes(32064)
        == 3072 * 8 * 32064 * 4 + runner.METADATA_ALLOWANCE
    )
    assert runner.estimated_bytes(151936) <= runner.MODELS["qwen"]["raw_limit_bytes"]
    assert runner.estimated_bytes(32064) <= runner.MODELS["phi"]["raw_limit_bytes"]


def test_model_pins_are_immutable_commits_and_no_runtime_model_choice():
    assert set(runner.MODELS) == {"qwen", "phi"}
    assert all(len(info["revision"]) == 40 for info in runner.MODELS.values())
    assert (
        runner.MODELS["phi"]["revision"] == "2fe192450127e6a83f7441aef6e3ca586c338b77"
    )


def test_all_eos_sources_are_used_despite_tokenizer_vocab_difference():
    model = SimpleNamespace(
        config=SimpleNamespace(eos_token_id=32007),
        generation_config=SimpleNamespace(eos_token_id=[32001, 32007]),
    )
    tokenizer = SimpleNamespace(eos_token_id=32000)
    assert runner.eos_ids(model, tokenizer) == {32000, 32001, 32007}


class FakeTokenizer:
    eos_token_id = 3

    def decode(self, token_ids, *, skip_special_tokens):
        assert not skip_special_tokens
        return "".join(
            {0: "red", 1: "blue", 2: " ", 3: ""}[token] for token in token_ids
        )


def test_answer_decode_removes_only_one_final_eos_and_preserves_other_specials():
    class SpecialTokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert not skip_special_tokens
            return "".join(
                {0: "red", 3: "<eos>", 4: "<user>"}[token] for token in token_ids
            )

    token_ids = [0, 4, 3]
    assert runner.decode_answer(SpecialTokenizer(), token_ids, {3}) == "red<user>"
    assert token_ids == [0, 4, 3]
    assert runner.decode_answer(SpecialTokenizer(), [3, 0], {3}) == "<eos>red"
    assert runner.decode_answer(SpecialTokenizer(), [0, 3, 3], {3}) == "red<eos>"


class FakeModel:
    config = SimpleNamespace(eos_token_id=3)
    generation_config = SimpleNamespace(eos_token_id=[3])

    def __init__(self):
        self.parameter = torch.zeros(1)
        self.calls = []

    def parameters(self):
        yield self.parameter

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        native = (
            torch.tensor([[[4.0, 1.0, 0.0, -1.0]]])
            if len(self.calls) == 1
            else torch.tensor([[[0.0, 1.0, 2.0, 4.0]]])
        )
        return SimpleNamespace(logits=native)


class FakeWriter:
    def __init__(self):
        self.saved = {}

    def add(self, identity, native):
        assert identity not in self.saved
        self.saved[identity] = native.clone()


def test_generation_retains_native_argmax_every_step_and_eos():
    model, writer = FakeModel(), FakeWriter()
    prediction = runner.new_prediction({"id": "world:probe", "expected_answer": "red"})
    counter = {"bytes": 0}
    runner.generate_row(
        model,
        FakeTokenizer(),
        {"input_ids": [8, 9]},
        prediction,
        writer,
        vocab_size=4,
        deadline=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        raw_limit=1000,
        counter=counter,
    )
    assert prediction["generated_ids"] == [0, 3]
    assert prediction["generated_text"] == "red"
    assert prediction["stop_reason"] == "eos"
    assert prediction["terminated"] and not prediction["truncated"]
    assert counter["bytes"] == 32
    assert len(writer.saved) == 2
    for token, endpoint in zip(
        prediction["generated_ids"], prediction["step_endpoint_ids"]
    ):
        assert int(writer.saved[endpoint].argmax()) == token
    assert all(
        call["use_cache"] is False and call["logits_to_keep"] == 1
        for call in model.calls
    )


def test_raw_ceiling_is_checked_before_endpoint_is_saved():
    writer, prediction = (
        FakeWriter(),
        runner.new_prediction({"id": "x", "expected_answer": "red"}),
    )
    with pytest.raises(RuntimeError, match="ceiling"):
        runner.generate_row(
            FakeModel(),
            FakeTokenizer(),
            {"input_ids": [8]},
            prediction,
            writer,
            vocab_size=4,
            deadline=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            raw_limit=15,
            counter={"bytes": 0},
        )
    assert writer.saved == {} and prediction["generated_ids"] == []


def test_newline_does_not_silently_stop_generation():
    class NewlineTokenizer(FakeTokenizer):
        def decode(self, token_ids, *, skip_special_tokens):
            return "\n" if 0 in token_ids else ""

    model, writer = FakeModel(), FakeWriter()
    prediction = runner.new_prediction({"id": "newline", "expected_answer": "red"})
    runner.generate_row(
        model,
        NewlineTokenizer(),
        {"input_ids": [8]},
        prediction,
        writer,
        vocab_size=4,
        deadline=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        raw_limit=1000,
        counter={"bytes": 0},
    )
    assert len(model.calls) == 2
    assert prediction["generated_ids"] == [0, 3]
    assert prediction["terminated"]


def test_no_eos_within_eight_steps_is_explicitly_truncated():
    class NeverEnds(FakeModel):
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[[4.0, 1.0, 0.0, -1.0]]]))

    prediction = runner.new_prediction({"id": "truncated", "expected_answer": "red"})
    runner.generate_row(
        NeverEnds(),
        FakeTokenizer(),
        {"input_ids": [8]},
        prediction,
        FakeWriter(),
        vocab_size=4,
        deadline=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        raw_limit=1000,
        counter={"bytes": 0},
    )
    assert len(prediction["generated_ids"]) == 8
    assert prediction["truncated"] and not prediction["terminated"]
    assert prediction["stop_reason"] == "max_tokens"


def test_checkpoint_paths_are_exclusive_and_preserve_every_completed_row(tmp_path):
    import json

    class Old:
        @staticmethod
        def seal_json(path, value):
            with Path(path).open("x") as handle:
                json.dump(value, handle)

    writer = SimpleNamespace(flush=lambda: None)
    records = [{"id": "a"}, {"id": "b"}]
    runner.checkpoint(Old(), writer, tmp_path, records)
    assert (
        json.loads((tmp_path / "progress-000002.json").read_text())["predictions"]
        == records
    )
    with pytest.raises(FileExistsError):
        runner.checkpoint(Old(), writer, tmp_path, records)


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "USER:" + messages[0]["content"] + "ASSISTANT:"

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        record = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            record["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return record


def token_row(identity, alias, *, field="color", alias_index=0):
    prefix = f"Read Alias {alias}.\n[ALIAS_END]\n"
    text = prefix + f"Return its {field}.\nAnswer:"
    marker = text.index("[ALIAS_END]")
    return {
        "id": identity,
        "world_id": "w",
        "split": "development",
        "text": text,
        "expected_answer": "red",
        "name_swap": 0,
        "focus_condition": "none",
        "query_alias_index": alias_index,
        "field": field,
        "alias_end_span": [marker, marker + len("[ALIAS_END]")],
        "field_start": len(prefix),
    }


def test_token_preparation_proves_shared_prefix_without_reading_attributes(monkeypatch):
    monkeypatch.setattr(runner, "DEVELOPMENT_COUNT", 2)
    rows = [token_row("a", "cat"), token_row("b", "cat", field="shape")]
    tokens = runner.tokenize_rows(CharacterTokenizer(), rows, 512)
    assert (
        tokens[0]["input_ids"][: tokens[0]["alias_end_position"] + 1]
        == tokens[1]["input_ids"][: tokens[1]["alias_end_position"] + 1]
    )


def test_token_preparation_rejects_future_field_leak(monkeypatch):
    monkeypatch.setattr(runner, "DEVELOPMENT_COUNT", 1)
    row = token_row("a", "cat")
    row["field_start"] = row["alias_end_span"][0]
    with pytest.raises(ValueError, match="forbidden future"):
        runner.tokenize_rows(CharacterTokenizer(), [row], 512)


def test_token_preparation_rejects_misaligned_coreferring_names(monkeypatch):
    monkeypatch.setattr(runner, "DEVELOPMENT_COUNT", 2)
    rows = [token_row("a", "cat"), token_row("b", "horse", alias_index=1)]
    with pytest.raises(ValueError, match="token geometry"):
        runner.tokenize_rows(CharacterTokenizer(), rows, 512)
