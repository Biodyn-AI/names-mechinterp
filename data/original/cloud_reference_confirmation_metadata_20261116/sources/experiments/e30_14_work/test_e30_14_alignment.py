"""Train-only maps and model-loader-independent causal mechanics."""

import io
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import e30_14_alignment as alignment


def training_problem():
    generator = torch.Generator().manual_seed(14)
    source = torch.randn(20, 4, dtype=torch.float64, generator=generator)
    transform = torch.linalg.qr(
        torch.randn(4, 4, dtype=torch.float64, generator=generator)
    ).Q
    target = source @ transform + torch.tensor([1.0, 2.0, 3.0, 4.0])
    ids = [f"train-{i}" for i in range(20)]
    return dict(zip(ids, source)), dict(zip(ids, target)), ids, transform


def fit_example(**kwargs):
    source, target, ids, transform = training_problem()
    fit = alignment.fit_alignment(
        source,
        target,
        training_entity_ids=ids,
        evaluation_entity_ids=[
            "validation-a",
            "validation-b",
            "test-a",
            "test-b",
            "test-c",
        ],
        rank=kwargs.pop("rank", 4),
        ridge_fraction=kwargs.pop("ridge_fraction", 1e-10),
        **kwargs,
    )
    return fit, transform


def test_ridge_recovers_unseen_difference_without_intercept_or_labels():
    fit, transform = fit_example()
    delta = torch.tensor([0.7, -0.3, 1.2, 0.5], dtype=torch.float64)
    predicted = alignment.apply_alignment(
        fit, delta, base_entity_id="test-a", donor_entity_id="test-b"
    )
    torch.testing.assert_close(predicted, delta @ transform, atol=1e-8, rtol=1e-8)
    assert fit["retained_rank"] == 4
    assert fit["training_entity_count"] == 20
    assert fit["training_rows_are_entities_not_ordered_pairs"]
    assert fit["ridge_alpha"] > 0
    assert fit["source_basis"].shape == (4, 4)
    assert fit["target_factor"].shape == (4, 4)


def test_centers_and_scale_are_training_only_and_offsets_cancel():
    source, target, ids, _ = training_problem()
    options = {
        "training_entity_ids": ids,
        "evaluation_entity_ids": ["test-a", "test-b"],
        "rank": 2,
        "ridge_fraction": 0.1,
    }
    first = alignment.fit_alignment(source, target, **options)
    shifted = alignment.fit_alignment(
        {key: value + 100 for key, value in source.items()},
        {key: value - 300 for key, value in target.items()},
        **options,
    )
    delta = torch.tensor([0.7, -0.3, 1.2, 0.5], dtype=torch.float64)
    a = alignment.apply_alignment(
        first, delta, base_entity_id="test-a", donor_entity_id="test-b"
    )
    b = alignment.apply_alignment(
        shifted, delta, base_entity_id="test-a", donor_entity_id="test-b"
    )
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_split_leakage_and_extra_vectors_fail_closed():
    source, target, ids, _ = training_problem()
    options = {
        "training_entity_ids": ids,
        "evaluation_entity_ids": ["test-a", "test-b"],
        "rank": 2,
        "ridge_fraction": 0.1,
    }
    with pytest.raises(ValueError, match="disjoint"):
        alignment.fit_alignment(
            source, target, **{**options, "evaluation_entity_ids": [ids[0], "test-a"]}
        )
    source["test-a"] = torch.full((4,), float("nan"))
    with pytest.raises(ValueError, match="exactly"):
        alignment.fit_alignment(source, target, **options)
    fit, _ = fit_example()
    for base, donor in (
        ("train-0", "test-a"),
        ("unknown", "test-a"),
        ("test-a", "test-a"),
    ):
        with pytest.raises(ValueError):
            alignment.apply_alignment(
                fit, torch.ones(4), base_entity_id=base, donor_entity_id=donor
            )


def test_permutation_null_fits_only_shuffled_training_correspondence():
    source, target, ids, _ = training_problem()
    options = {
        "training_entity_ids": ids,
        "evaluation_entity_ids": ["test-a", "test-b"],
        "rank": 4,
        "ridge_fraction": 0.1,
    }
    permutation = ids[1:] + ids[:1]
    fit = alignment.fit_alignment(
        source, target, target_permutation=permutation, **options
    )
    explicit = alignment.fit_alignment(
        source, dict(zip(ids, [target[key] for key in permutation])), **options
    )
    torch.testing.assert_close(
        fit["source_basis"] @ fit["target_factor"],
        explicit["source_basis"] @ explicit["target_factor"],
    )
    assert fit["training_target_entity_order"] == permutation
    with pytest.raises(ValueError, match="permutation"):
        alignment.fit_alignment(
            source, target, target_permutation=ids[:-1] + [ids[0]], **options
        )


def test_partial_procrustes_stays_in_identified_training_subspace():
    fit, transform = fit_example(method="orthogonal", ridge_fraction=0)
    delta = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(
        alignment.apply_alignment(
            fit, delta, base_entity_id="test-a", donor_entity_id="test-b"
        ),
        delta @ transform,
        atol=1e-10,
        rtol=1e-10,
    )
    assert fit["arbitrary_ambient_nullspace_completion"] is False
    assert fit["method"] == "orthogonal"


def test_controls_have_declared_doses_and_reproducible_directions():
    fit, _ = fit_example()
    arguments = {
        "fit": fit,
        "source_delta": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "base_entity_id": "test-a",
        "donor_entity_id": "test-b",
        "wrong_source_delta": torch.tensor([4.0, 1.0, -2.0, 3.0]),
        "wrong_donor_entity_id": "test-c",
        "same_form_delta": torch.tensor([0.2, 0.3, 0.4, 0.5]),
        "random_seed": 14,
    }
    primary = alignment.make_intervention("aligned", **arguments)
    for condition in ("random", "wrong_donor", "reverse"):
        record = alignment.make_intervention(condition, **arguments)
        assert record["actual_norm"] == pytest.approx(primary["actual_norm"], rel=1e-12)
    first = alignment.make_intervention("random", **arguments)
    second = alignment.make_intervention("random", **arguments)
    torch.testing.assert_close(first["delta"], second["delta"], atol=0, rtol=0)
    torch.testing.assert_close(
        alignment.make_intervention("reverse", **arguments)["delta"], -primary["delta"]
    )
    torch.testing.assert_close(
        alignment.make_intervention("unmapped", **arguments)["delta"],
        arguments["source_delta"].double(),
    )
    doubled = alignment.make_intervention("aligned", gain=2, **arguments)
    torch.testing.assert_close(doubled["delta"], 2 * primary["delta"])


def test_validation_helper_does_not_accept_test_vectors_as_extra_rows():
    fit, transform = fit_example()
    source = {
        "validation-a": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "validation-b": torch.tensor([4.0, 3.0, 2.0, 1.0]),
    }
    target = {key: value.double() @ transform for key, value in source.items()}
    result = alignment.score_validation(
        fit,
        source,
        target,
        validation_entity_ids=list(source),
        test_entity_ids=["test-a", "test-b", "test-c"],
    )
    assert result["relative_squared_error"] < 1e-15
    assert result["ordered_pair_count"] == 2
    source["test-a"] = torch.ones(4)
    with pytest.raises(ValueError, match="exactly"):
        alignment.score_validation(
            fit,
            source,
            target,
            validation_entity_ids=["validation-a", "validation-b"],
            test_entity_ids=["test-a", "test-b", "test-c"],
        )


class TinyBlock(nn.Module):
    def __init__(self, tuple_output):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.tuple_output = tuple_output

    def forward(self, hidden):
        value = hidden + self.linear(hidden.cumsum(1)).tanh()
        return (value, "tail") if self.tuple_output else value


class TinyModel(nn.Module):
    def __init__(self, tuple_output):
        super().__init__()
        generator = torch.Generator().manual_seed(14)
        self.embedding = nn.Embedding(32, 4)
        self.blocks = nn.ModuleList([TinyBlock(tuple_output) for _ in range(2)])
        self.head = nn.Linear(4, 32, bias=False)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) / 4)
        self.double().eval().requires_grad_(False)
        self.forward_calls = 0
        self.fail = False

    def forward(
        self, *, input_ids, attention_mask, use_cache, return_dict, logits_to_keep
    ):
        assert input_ids.shape[0] == 1 and use_cache is False and return_dict is True
        assert logits_to_keep == 1 and bool(attention_mask.all())
        self.forward_calls += 1
        hidden = self.embedding(input_ids)
        for block in self.blocks:
            result = block(hidden)
            if isinstance(result, tuple):
                assert result[1] == "tail"
                hidden = result[0]
            else:
                hidden = result
        if self.fail:
            raise RuntimeError("synthetic failure")
        return SimpleNamespace(logits=self.head(hidden[:, -1:]))


def token_prompts():
    # The marker follows the WHOLE name, whose token length differs by row.
    return [
        {
            "prompt_id": "a",
            "input_ids": [1, 2, 9, 5],
            "marker_position": 2,
            "query_position": 3,
            "marker_token_id": 9,
        },
        {
            "prompt_id": "b",
            "input_ids": [1, 3, 4, 9, 5],
            "marker_position": 3,
            "query_position": 4,
            "marker_token_id": 9,
        },
    ]


@pytest.mark.parametrize("tuple_output", [True, False])
def test_capture_variable_name_lengths_and_oracles_precede_causal_patches(tuple_output):
    model = TinyModel(tuple_output)
    executor = alignment.ResidualPatchExecutor(
        model, list(model.blocks), control_tolerance=1e-10
    )
    captured = executor.capture_prompts(token_prompts())
    assert captured["physical_forward_count"] == 2
    assert set(captured["states"]["a"]) == {
        "residual::0::marker",
        "residual::0::query",
        "residual::1::marker",
        "residual::1::query",
    }
    with pytest.raises(RuntimeError, match="oracle"):
        executor.run_patch("a", layer_index=0, token_site="marker", delta=torch.ones(4))
    qualification = executor.qualify_oracles(
        sites=[(0, "marker"), (0, "query")],
        donor_pairs=[("a", "b"), ("b", "a")],
    )
    assert qualification["passed"]
    assert qualification["self_patch_max_abs_error"] == 0
    assert qualification["final_query_matched_max_abs_error"] == 0
    changed = executor.run_patch(
        "a", layer_index=0, token_site="marker", delta=torch.ones(4)
    )
    assert not torch.equal(changed, captured["clean_logits"]["a"])
    assert all(not module._forward_hooks for module in model.modules())


def test_patch_failure_removes_hooks_and_model_changes_are_rejected():
    model = TinyModel(False)
    executor = alignment.ResidualPatchExecutor(
        model, list(model.blocks), control_tolerance=1e-10
    )
    executor.capture_prompts(token_prompts())
    executor.qualify_oracles(sites=[(0, "marker")])
    model.fail = True
    with pytest.raises(RuntimeError, match="synthetic failure"):
        executor.run_patch("a", layer_index=0, token_site="marker", delta=torch.ones(4))
    assert all(not module._forward_hooks for module in model.modules())
    model.fail = False
    with torch.no_grad():
        model.head.weight.add_(0.01)
    with pytest.raises(RuntimeError, match="parameters changed"):
        executor.run_patch("a", layer_index=0, token_site="marker", delta=torch.ones(4))


def test_unknown_marker_or_unqualified_site_fails_closed():
    model = TinyModel(False)
    executor = alignment.ResidualPatchExecutor(model, list(model.blocks))
    bad = deepcopy(token_prompts())
    bad[1]["marker_token_id"] = 3
    with pytest.raises(ValueError, match="marker"):
        executor.capture_prompts(bad)


@pytest.mark.parametrize("architecture", ["gpt2", "qwen2"])
def test_actual_untrained_decoder_contracts_with_final_only_head(architecture):
    transformers = pytest.importorskip("transformers")
    with torch.random.fork_rng():
        torch.manual_seed(14)
        if architecture == "gpt2":
            config = transformers.GPT2Config(
                vocab_size=32,
                n_positions=16,
                n_embd=8,
                n_layer=2,
                n_head=2,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
            )
            config._attn_implementation = "eager"
            model = transformers.GPT2LMHeadModel(config)
            blocks = list(model.transformer.h)
        else:
            config = transformers.Qwen2Config(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=16,
                attention_dropout=0,
            )
            config._attn_implementation = "eager"
            model = transformers.Qwen2ForCausalLM(config)
            blocks = list(model.model.layers)
    model.double().eval().requires_grad_(False)
    executor = alignment.ResidualPatchExecutor(model, blocks, control_tolerance=1e-10)
    captured = executor.capture_prompts(token_prompts())
    oracles = executor.qualify_oracles(sites=[(0, "marker"), (1, "query")])
    assert oracles["self_patch_max_abs_error"] == 0
    assert oracles["final_query_matched_max_abs_error"] == 0
    assert captured["clean_logits"]["a"].shape == (32,)


def test_permutation_dose_and_zero_controls_are_explicit():
    source, target, ids, _ = training_problem()
    options = {
        "training_entity_ids": ids,
        "evaluation_entity_ids": ["test-a", "test-b"],
        "rank": 4,
        "ridge_fraction": 0.1,
    }
    real = alignment.fit_alignment(source, target, **options)
    shuffled = alignment.fit_alignment(
        source, target, target_permutation=ids[1:] + ids[:1], **options
    )
    arguments = {
        "fit": real,
        "source_delta": torch.ones(4),
        "base_entity_id": "test-a",
        "donor_entity_id": "test-b",
        "permutation_fit": shuffled,
    }
    aligned = alignment.make_intervention("aligned", **arguments)
    permuted = alignment.make_intervention("permuted_dosematched", **arguments)
    assert aligned["actual_norm"] == pytest.approx(permuted["actual_norm"], rel=1e-12)
    zero = alignment.make_intervention(
        "random", **{**arguments, "source_delta": torch.zeros(4)}
    )
    assert zero["degenerate_control"] and zero["actual_norm"] == 0
    separately_selected = alignment.fit_alignment(
        source,
        target,
        target_permutation=ids[1:] + ids[:1],
        **{**options, "rank": 2, "ridge_fraction": 1.0},
    )
    selected_control = alignment.make_intervention(
        "permuted_dosematched", **{**arguments, "permutation_fit": separately_selected}
    )
    assert selected_control["actual_norm"] == pytest.approx(
        aligned["actual_norm"], rel=1e-12
    )
    changed = deepcopy(shuffled)
    changed["source_training_matrix_sha256"] = "different-training-data"
    with pytest.raises(ValueError, match="training data"):
        alignment.make_intervention(
            "permuted_dosematched", **{**arguments, "permutation_fit": changed}
        )


def test_failed_capture_cannot_be_qualified_afterward():
    model = TinyModel(False)
    executor = alignment.ResidualPatchExecutor(model, list(model.blocks))
    model.fail = True
    with pytest.raises(RuntimeError, match="synthetic failure"):
        executor.capture_prompts(token_prompts())
    model.fail = False
    with pytest.raises(RuntimeError, match="capture"):
        executor.qualify_oracles(sites=[(0, "marker")])


def test_map_records_are_weights_only_serializable_and_fingerprinted():
    fit, _ = fit_example()
    stream = io.BytesIO()
    torch.save(fit, stream)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=True)
    assert alignment.alignment_digest(loaded) == fit["alignment_sha256"]
    loaded["source_basis"][0, 0] += 0.01
    assert alignment.alignment_digest(loaded) != fit["alignment_sha256"]


def test_qualification_does_not_cross_product_separately_qualified_panels():
    model = TinyModel(False)
    executor = alignment.ResidualPatchExecutor(
        model, list(model.blocks), control_tolerance=1e-10
    )
    prompts = token_prompts()
    extra = deepcopy(prompts)
    extra[0]["prompt_id"], extra[1]["prompt_id"] = "c", "d"
    executor.capture_prompts(prompts + extra)
    executor.qualify_oracles(sites=[(0, "marker")], prompt_ids=["a", "b"])
    executor.qualify_oracles(sites=[(0, "query")], prompt_ids=["c", "d"])
    with pytest.raises(RuntimeError, match="oracle"):
        executor.run_patch("a", layer_index=0, token_site="query", delta=torch.ones(4))
    with pytest.raises(RuntimeError, match="oracle"):
        executor.run_patch("c", layer_index=0, token_site="marker", delta=torch.ones(4))


def test_oracle_failure_is_terminal_and_reports_both_errors():
    model = TinyModel(False)
    executor = alignment.ResidualPatchExecutor(
        model, list(model.blocks), control_tolerance=1e-10
    )
    executor.capture_prompts(token_prompts())
    executor._clean_logits["a"] = executor._clean_logits["a"] + 0.01
    with pytest.raises(RuntimeError, match="self=.*final_query=.*tolerance="):
        executor.qualify_oracles(sites=[(0, "marker")])
    with pytest.raises(RuntimeError, match="cannot be resumed"):
        executor.qualify_oracles(sites=[(0, "marker")])
    assert all(not module._forward_hooks for module in model.modules())
