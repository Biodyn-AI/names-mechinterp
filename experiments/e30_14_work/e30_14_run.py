"""One-shot, staged E30.14 runner. No resource provisioning or credential access.

prepare freezes design/source/facts/pins before downloading public token assets,
then freezes model-free tokens and the combined storage plan before any weights.
run retains one model through qualification, fit/validation, and held-out work.
Every failed output namespace remains consumed; neither resume nor overwrite is
supported. rescore checks saved native endpoints without loading any model.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import string
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
DEADLINE = "2026-11-10T20:30:00Z"
PYTHON = "/workspace/runtime/e30_13/bin/python"
CACHE_ROOT = "/workspace/models/hub"
MODELS = {
    "medium": {
        "model_id": "openai-community/gpt2-medium",
        "revision": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        "chat_model": False,
    },
    "modern": {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "chat_model": True,
    },
}
REQUIRED_SOURCES = (
    "e30_14_run.py",
    "e30_14_protocol.py",
    "e30_14_alignment.py",
    "e30_14_capture.py",
    "e30_14_evaluate.py",
    "e30_14_summary.py",
)
TOKEN_ASSETS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
)


def modules():
    return SimpleNamespace(
        **{
            name: importlib.import_module("e30_14_" + name)
            for name in ("protocol", "capture", "alignment", "evaluate", "summary")
        }
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_bytes(path, content):
    path = Path(path)
    with path.open("xb") as handle:
        handle.write(content)
    digest = sha256(path)
    with path.with_name(path.name + ".sha256").open("x") as handle:
        handle.write(digest + "\n")
    return digest


def seal_json(path, record):
    return write_bytes(
        path,
        (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
    )


def verify_file(path, expected=None):
    path = Path(path)
    expected = (
        path.with_name(path.name + ".sha256").read_text().strip()
        if expected is None
        else expected
    )
    if sha256(path) != expected:
        raise ValueError("artifact checksum mismatch")
    return expected


def save_pt(path, record):
    import torch

    path = Path(path)
    with path.open("xb") as handle:
        torch.save(record, handle)
    digest = sha256(path)
    with path.with_name(path.name + ".sha256").open("x") as handle:
        handle.write(digest + "\n")
    return digest


def check_deadline(deadline):
    if datetime.now(UTC) >= datetime.fromisoformat(deadline):
        raise RuntimeError("registered execution deadline has passed")


def source_hashes(root):
    paths = {path.name: path for path in Path(root).glob("e30_14*.py")}
    if set(REQUIRED_SOURCES) - set(paths):
        raise ValueError("required runtime source closure is incomplete")
    paths["us_state_facts_2020.json"] = Path(root) / "us_state_facts_2020.json"
    return {name: sha256(path) for name, path in sorted(paths.items())}


def load_tokenizer(snapshot):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(snapshot),
        use_fast=True,
        local_files_only=True,
        token=False,
        trust_remote_code=False,
    )


def snapshot_file(path, snapshot):
    """Allow repository blobs and HF's shared public blobs, never cache escapes."""
    snapshot, path = Path(snapshot), Path(path)
    resolved = path.resolve(strict=True)
    cache_root = snapshot.parent.parent.parent.resolve()
    if not resolved.is_relative_to(cache_root):
        raise ValueError("snapshot file escaped its public model cache")
    allowed = {snapshot.resolve(), (snapshot.parent.parent / "blobs").resolve()}
    parts = resolved.relative_to(cache_root).parts
    # Recent Hub versions add a second link into cache/blobs/xx/<64 hex key>.
    # Match that precise shape; do not admit arbitrary files under the cache.
    shared_blob = (
        len(parts) == 3
        and parts[0] == "blobs"
        and len(parts[1]) == 2
        and len(parts[2]) == 64
        and parts[2].startswith(parts[1])
        and all(character in "0123456789abcdef" for character in parts[2])
    )
    if (
        path.parent != snapshot
        or not (resolved.parent in allowed or shared_blob)
        or not resolved.is_file()
    ):
        raise ValueError("snapshot file escaped its public model cache")
    return resolved


def load_token_assets(info, *, cache_root=CACHE_ROOT):
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            info["model_id"],
            revision=info["revision"],
            token=False,
            allow_patterns=list(TOKEN_ASSETS),
            cache_dir=str(cache_root),
        )
    )
    hashes = {
        name: sha256(snapshot_file(snapshot / name, snapshot))
        for name in TOKEN_ASSETS
        if (snapshot / name).is_file()
    }
    return {
        "snapshot": str(snapshot),
        "tokenizer": load_tokenizer(snapshot),
        "config": read_json(snapshot_file(snapshot / "config.json", snapshot)),
        "asset_sha256": hashes,
    }


def prepare(output, *, source_root=HERE, cache_root=CACHE_ROOT):
    output, source_root = Path(output), Path(source_root)
    output.mkdir()  # Exclusive namespace; an existing failed attempt is never reused.
    try:
        api = modules()
        sources = source_hashes(source_root)
        (output / "sources").mkdir()
        for name in sources:
            write_bytes(output / "sources" / name, (source_root / name).read_bytes())
        seal_json(output / "SOURCES.json", sources)
        design = api.protocol.build_protocol()
        seal_json(output / "DESIGN.json", design)
        write_bytes(
            output / "FACTS.json",
            (source_root / "us_state_facts_2020.json").read_bytes(),
        )
        seal_json(output / "MODELS.json", MODELS)
        facts, configs = read_json(output / "FACTS.json"), {}
        for name, info in MODELS.items():
            assets = load_token_assets(info, cache_root=cache_root)
            rows = api.capture.prepare_token_rows(
                assets["tokenizer"],
                api.protocol.build_prompts(),
                chat_model=info["chat_model"],
                facts=facts["facts"],
            )
            qualification = api.capture.prepare_token_rows(
                assets["tokenizer"],
                api.protocol.build_qualification_prompts(facts),
                chat_model=info["chat_model"],
                facts=facts["facts"],
            )
            if len(rows) != 224 or len(qualification) != 32:
                raise ValueError("registered prompt counts changed")
            record = api.capture.make_token_manifest(
                rows, facts, model_id=info["model_id"], model_config=assets["config"]
            )
            record.update(
                qualification_rows=qualification,
                snapshot=assets["snapshot"],
                asset_sha256=assets["asset_sha256"],
                revision=info["revision"],
                cache_root=str(Path(cache_root).absolute()),
            )
            seal_json(output / f"TOKENS-{name}.json", record)
            configs[info["model_id"]] = assets["config"]
        storage = api.capture.estimate_storage(configs, protocol_record=design)
        seal_json(output / "STORAGE.json", storage)
        if source_hashes(source_root) != sources:
            raise ValueError("runtime source changed during preparation")
        inputs = [
            "SOURCES.json",
            "DESIGN.json",
            "FACTS.json",
            "MODELS.json",
            "STORAGE.json",
            *[f"TOKENS-{name}.json" for name in MODELS],
        ]
        registration = {
            "schema_version": "e30_14_final_registration_v1",
            "models": MODELS,
            "input_sha256": {name: verify_file(output / name) for name in inputs},
            "runtime_python": PYTHON,
            "python_version": sys.version,
            "deadline_utc": DEADLINE,
            "runtime_versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "tokenizers",
                    "numpy",
                    "safetensors",
                    "huggingface-hub",
                )
            },
            "execution_scope": {
                "authorized_by_current_user_request": True,
                "tokenizer_qualification_authorized": True,
                "model_execution_authorized": True,
                "new_paid_resource_creation_authorized": False,
                "combined_raw_storage_limit_bytes": 15000000000,
                "dtype": "float32",
                "device": "cuda:0",
                "attention": "eager",
                "prompt_batch_size": 1,
                "lm_head_positions": 1,
                "tf32": False,
                "cpu_threads": 1,
                "one_attempt_per_model": True,
                "no_resume_or_failure_based_rescue": True,
            },
        }
        seal_json(output / "REGISTRATION.json", registration)
        return registration
    except BaseException:
        seal_json(
            output / "failure.json",
            {"phase": "prepare", "traceback": traceback.format_exc()},
        )
        raise


def check_registration(output, *, source_root=HERE):
    output = Path(output)
    verify_file(output / "REGISTRATION.json")
    registration = read_json(output / "REGISTRATION.json")
    if (
        registration["models"] != MODELS
        or not registration["execution_scope"]["model_execution_authorized"]
    ):
        raise ValueError("registration model pins or execution scope changed")
    for name, digest in registration["input_sha256"].items():
        verify_file(output / name, digest)
    sources = read_json(output / "SOURCES.json")
    if source_hashes(source_root) != sources:
        raise ValueError("runtime source checksum mismatch")
    for name, digest in sources.items():
        verify_file(output / "sources" / name, digest)
    for name in MODELS:
        tokens = read_json(output / f"TOKENS-{name}.json")
        for asset, digest in tokens["asset_sha256"].items():
            verify_file(
                snapshot_file(Path(tokens["snapshot"]) / asset, tokens["snapshot"]),
                digest,
            )
    storage = read_json(output / "STORAGE.json")
    if not storage["within_limit"] or storage["estimated_total_bytes"] > 15000000000:
        raise ValueError("combined storage gate failed")
    return registration


def configure_runtime(registration):
    import os

    import torch

    if str(Path(sys.executable).absolute()) != registration["runtime_python"]:
        raise RuntimeError("use the exact registered runtime Python")
    if sys.version != registration["python_version"]:
        raise RuntimeError("registered Python version changed")
    for name, version in registration["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("registered runtime package version changed")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("registered CUDA runtime is unavailable")
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "dtype": "float32",
        "attention": "eager",
        "batch_size": 1,
        "lm_head_positions": 1,
        "tf32": False,
        "cpu_threads": 1,
    }


def load_model(info, tokens, model_files_path):
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        info["model_id"],
        revision=info["revision"],
        token=False,
        allow_patterns=[*TOKEN_ASSETS, "*.safetensors", "*.safetensors.index.json"],
        cache_dir=tokens["cache_root"],
    )
    if Path(snapshot).resolve() != Path(tokens["snapshot"]).resolve():
        raise ValueError("weight snapshot differs from the registered token snapshot")
    for asset, digest in tokens["asset_sha256"].items():
        verify_file(snapshot_file(Path(snapshot) / asset, snapshot), digest)
    paths = [
        path
        for path in Path(snapshot).iterdir()
        if path.name in TOKEN_ASSETS
        or path.name.endswith((".safetensors", ".safetensors.index.json"))
    ]
    if not any(path.name.endswith(".safetensors") for path in paths):
        raise ValueError("pinned snapshot has no safetensors weights")
    for path in paths:
        if path.name.endswith(".safetensors.index.json"):
            referenced = set(
                read_json(snapshot_file(path, snapshot))["weight_map"].values()
            )
            if any(
                Path(name).name != name or name not in {item.name for item in paths}
                for name in referenced
            ):
                raise ValueError("weight index escaped the pinned snapshot")
    fingerprints, files = {}, {}
    for path in sorted(paths):
        actual = snapshot_file(path, snapshot)
        metadata = actual.stat()
        fingerprints[path.name] = (
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        files[path.name] = {"sha256": sha256(actual), "size_bytes": metadata.st_size}
    seal_json(
        model_files_path,
        {"model_id": info["model_id"], "revision": info["revision"], "files": files},
    )
    model = modules().capture.load_local_model(snapshot, info["model_id"])
    for path in paths:
        metadata = snapshot_file(path, snapshot).stat()
        if (
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ) != fingerprints[path.name]:
            raise RuntimeError("pinned model file changed while loading")
    return model


def normalize_city(value):
    return (
        " ".join(value.strip().casefold().split()).rstrip(string.punctuation).rstrip()
    )


def qualify_model(model, tokenizer, rows, facts, *, deadline):
    import torch

    from e30_14_evaluate import js_divergence

    results, first_logits, calls = [], {}, 0
    eos = model.config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    eos.add(tokenizer.eos_token_id)
    device = next(model.parameters()).device
    for row in rows:
        generated = []
        for _ in range(12):
            check_deadline(deadline)
            ids = torch.tensor(
                [row["input_ids"] + generated], device=device, dtype=torch.long
            )
            with torch.inference_mode():
                logits = model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=1,
                ).logits
            calls += 1
            if (
                logits.dtype != torch.float32
                or logits.shape[:2] != (1, 1)
                or not bool(torch.isfinite(logits).all())
            ):
                raise RuntimeError(
                    "qualification requires finite native float32 B1/head1 logits"
                )
            if not generated:
                first_logits[row["prompt_id"]] = logits[0, 0].detach().cpu().clone()
            token = int(logits[0, 0].argmax())
            generated.append(token)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            if token in eos or "\n" in text:
                break
        city = text.split("\n", 1)[0]
        aliases = facts["answer_aliases"][row["entity_id"]][row["relation_id"]]
        results.append(
            {
                **row,
                "generated_ids": generated,
                "generated_text": text,
                "city_text": city,
                "correct": normalize_city(city)
                in {normalize_city(alias) for alias in aliases},
            }
        )
    nearest = {}
    for relation in ("capital", "largest_city"):
        comparisons = []
        for row in rows:
            if row["relation_id"] != relation or row["form"] != "abbr":
                continue
            full = [
                candidate
                for candidate in rows
                if candidate["relation_id"] == relation
                and candidate["form"] == "full"
                and candidate["frame"] == row["frame"]
            ]
            distances = {
                candidate["entity_id"]: js_divergence(
                    first_logits[row["prompt_id"]], first_logits[candidate["prompt_id"]]
                )
                for candidate in full
            }
            winners = [
                entity
                for entity, value in distances.items()
                if value == min(distances.values())
            ]
            comparisons.append(
                {
                    "prompt_id": row["prompt_id"],
                    "distances": distances,
                    "correct": winners == [row["entity_id"]],
                }
            )
        nearest[relation] = comparisons
    groups = {
        f"{relation}::{form}": sum(
            row["correct"]
            for row in results
            if row["relation_id"] == relation and row["form"] == form
        )
        for relation in ("capital", "largest_city")
        for form in ("full", "abbr")
    }
    correct = sum(row["correct"] for row in results)
    passed = (
        len(rows) == 32
        and correct >= 28
        and all(value >= 7 for value in groups.values())
        and len(nearest["capital"]) == 8
        and all(row["correct"] for row in nearest["capital"])
    )
    return {
        "passed": passed,
        "correct": correct,
        "total": len(rows),
        "relation_form_correct": groups,
        "nearest_output": nearest,
        "physical_forward_count": calls,
        "rows": results,
    }, first_logits


class RawWriter:
    """Bounded shards retain every native endpoint, not just derived scores."""

    def __init__(self, root, shard_size=128):
        self.root, self.shard_size, self.pending = Path(root), shard_size, {}
        self.index = {
            "schema_version": "e30_14_native_float32_v1",
            "endpoints": {},
            "shards": [],
        }
        self.closed = False

    def add(self, identity, value):
        import torch

        if (
            value.dtype != torch.float32
            or value.ndim != 1
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("raw endpoints must be finite native float32 vectors")
        if (
            self.closed
            or identity in self.index["endpoints"]
            or identity in self.pending
        ):
            raise ValueError("raw endpoint identity reused or writer closed")
        self.pending[identity] = value.detach().cpu().contiguous().clone()
        if len(self.pending) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        from safetensors.torch import save

        name = f"raw-{len(self.index['shards']):04d}.safetensors"
        digest = write_bytes(self.root / name, save(self.pending))
        self.index["shards"].append(
            {"file": name, "sha256": digest, "endpoint_count": len(self.pending)}
        )
        for identity, value in self.pending.items():
            self.index["endpoints"][identity] = {
                "file": name,
                "key": identity,
                "dtype": "float32",
                "shape": list(value.shape),
            }
        self.pending.clear()

    def finish(self):
        self.flush()
        self.closed = True
        seal_json(self.root / "RAW_INDEX.json", self.index)
        return self.index


class RawReader:
    def __init__(self, root, index):
        self.root, self.index, self.verified = Path(root), index, set()
        self.digests = {shard["file"]: shard["sha256"] for shard in index["shards"]}

    def read(self, identity):
        from safetensors import safe_open

        item = self.index["endpoints"][identity]
        if item["file"] not in self.verified:
            verify_file(self.root / item["file"], self.digests[item["file"]])
            self.verified.add(item["file"])
        with safe_open(
            self.root / item["file"], framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(item["key"])


def score_saved_cells(reader, cells, model_id):
    import torch

    torch.set_num_threads(1)
    evaluate = modules().evaluate
    return [
        {
            **cell,
            "model_id": model_id,
            "direction": cell["source_form"] + "_to_" + cell["target_form"],
            "scores": evaluate.score_cell(
                {
                    name: reader.read(identity)
                    for name, identity in cell["conditions"].items()
                }
            ),
        }
        for cell in cells
    ]


def clean_fact_diagnostics(rows, clean_logits, tokenizer, facts):
    """Descriptive canonical FIRST-TOKEN knowledge only; never filters cases."""
    records = []
    for row in rows:
        logits, expected = (
            clean_logits[row["prompt_id"]].double(),
            row["answer_first_token_id"],
        )
        top = int(logits.argmax())
        others = {
            item["answer_first_token_id"]
            for item in rows
            if item["relation_id"] == row["relation_id"]
            and item["entity_id"] != row["entity_id"]
        }
        records.append(
            {
                "prompt_id": row["prompt_id"],
                "entity_id": row["entity_id"],
                "relation_id": row["relation_id"],
                "form": row["form"],
                "frame": row["frame"],
                "canonical_answer": facts["facts"][row["entity_id"]][
                    row["relation_id"]
                ],
                "expected_first_token_id": expected,
                "expected_rank": 1 + int((logits > logits[expected]).sum()),
                "expected_probability": float(logits.log_softmax(-1)[expected].exp()),
                "argmax_id": top,
                "argmax_decoded_token": tokenizer.decode(
                    [top], skip_special_tokens=False
                ),
                "canonical_first_token_top1": top == expected,
                "margin_vs_other_state_first_tokens": None
                if not others
                else float(logits[expected] - logits[sorted(others)].max()),
                "other_state_first_token_collision": expected in others,
            }
        )

    def aggregate(values):
        return {
            "count": len(values),
            "canonical_first_token_top1_count": sum(
                item["canonical_first_token_top1"] for item in values
            ),
            "mean_expected_rank": sum(item["expected_rank"] for item in values)
            / len(values),
        }

    groups = {
        "::".join((item["relation_id"], item["form"], item["frame"]))
        for item in records
    }
    return {
        "first_token_only_not_whole_city_accuracy": True,
        "no_filtering_or_model_rescue": True,
        "rows": records,
        "global": aggregate(records),
        "by_relation_form_frame": {
            group: aggregate(
                [
                    item
                    for item in records
                    if "::".join((item["relation_id"], item["form"], item["frame"]))
                    == group
                ]
            )
            for group in sorted(groups)
        },
    }


def seal_custody(root, output, counts):
    artifacts = {
        path.name: sha256(path)
        for path in root.iterdir()
        if path.is_file() and not path.name.endswith(".sha256")
    }
    record = {
        "artifact_sha256": artifacts,
        "registration_sha256": verify_file(output / "REGISTRATION.json"),
        "physical_counts": counts,
        "source_sha256": read_json(output / "SOURCES.json"),
        "registered_input_sha256": read_json(output / "REGISTRATION.json")[
            "input_sha256"
        ],
    }
    seal_json(root / "CUSTODY.json", record)


def run_model(output, model_name, *, source_root=HERE):
    if model_name not in MODELS:
        raise ValueError("model is outside the registered panel")
    output = Path(output)
    registration = check_registration(output, source_root=source_root)
    root = output / model_name
    root.mkdir()
    fit_executor = test_executor = model = None
    counts = {}
    try:
        info, api = MODELS[model_name], modules()
        check_deadline(registration["deadline_utc"])
        seal_json(root / "RUNTIME.json", configure_runtime(registration))
        tokens, facts, design = (
            read_json(output / f"TOKENS-{model_name}.json"),
            read_json(output / "FACTS.json"),
            read_json(output / "DESIGN.json"),
        )
        model = load_model(info, tokens, root / "MODEL_FILES.json")
        tokenizer = load_tokenizer(tokens["snapshot"])
        qualification, qualification_logits = qualify_model(
            model,
            tokenizer,
            tokens["qualification_rows"],
            facts,
            deadline=registration["deadline_utc"],
        )
        counts["qualification"] = qualification["physical_forward_count"]
        save_pt(root / "QUALIFICATION.pt", qualification_logits)
        qualification["interpretation"] = (
            "computational bridge, not behavior-qualified semantics"
            if model_name == "medium"
            else "preselected semantic model; no rescue on failure"
        )
        seal_json(root / "QUALIFICATION.json", qualification)
        if model_name == "modern" and not qualification["passed"]:
            result = {
                "status": "semantic_qualification_failed",
                "held_out_capture_performed": False,
            }
            seal_json(root / "RESULT.json", result)
            seal_custody(root, output, counts)
            return result
        check_deadline(registration["deadline_utc"])
        training_rows = [
            row for row in tokens["rows"] if row["split"] in ("fit", "validation")
        ]
        fit_executor, training = api.capture.capture_stage(
            model, training_rows, model_id=info["model_id"]
        )
        counts["fit_validation"] = fit_executor.physical_forward_count
        save_pt(root / "FIT_VALIDATION.pt", training)
        selected = api.capture.select_maps(
            training["states"], training_rows, protocol_record=design
        )
        maps_hash = save_pt(root / "MAPS.pt", selected)
        selection = api.capture.selection_metadata(selected)
        selection["maps_file_sha256"] = maps_hash
        seal_json(root / "SELECTION.json", selection)
        fit_executor.close()
        fit_executor = None
        check_registration(output, source_root=source_root)
        check_deadline(registration["deadline_utc"])
        test_executor = api.alignment.ResidualPatchExecutor(
            model, api.capture.model_blocks(model, info["model_id"])
        )
        test_rows = [row for row in tokens["rows"] if row["split"] == "test"]
        testing = test_executor.capture_prompts(test_rows)
        counts["test_clean"] = test_executor.physical_forward_count
        save_pt(root / "TEST_CLEAN.pt", testing)
        seal_json(
            root / "HELDOUT_CLEAN_FACT_DIAGNOSTICS.json",
            clean_fact_diagnostics(
                test_rows, testing["clean_logits"], tokenizer, facts
            ),
        )
        target_ids = [
            row["prompt_id"]
            for row in tokens["rows"]
            if row["split"] == "test" and row["frame"] == "C"
        ]
        cells = tokens["comparison_cells"]
        pairs = sorted(
            {(cell["base_prompt_id"], cell["donor_prompt_id"]) for cell in cells}
        )
        sites = [
            (item["layer_index"], "marker" if family == "entity_marker" else "query")
            for family, item in selected.items()
        ]
        oracle = test_executor.qualify_oracles(
            sites=sites, prompt_ids=target_ids, donor_pairs=pairs
        )
        seal_json(root / "ORACLES.json", oracle)
        if not oracle["passed"]:
            raise RuntimeError("mathematical oracle gate failed")
        counts["oracles"] = oracle["physical_forward_count"]
        jobs, logical = api.evaluate.build_jobs(
            cells, tokens["rows"], testing["states"], selected
        )
        plan_hash = save_pt(root / "PLAN.pt", {"jobs": jobs, "cells": logical})
        plan = {
            "jobs": [
                {key: value for key, value in job.items() if key != "operand"}
                for job in jobs
            ],
            "cells": logical,
            "operand_file_sha256": plan_hash,
            "plan_sha256": api.evaluate.plan_digest(jobs, logical),
        }
        seal_json(root / "PLAN.json", plan)
        writer = RawWriter(root)
        for identity, endpoint in testing["clean_logits"].items():
            writer.add("clean::" + identity, endpoint)
        for job in jobs:
            check_deadline(registration["deadline_utc"])
            endpoint = test_executor.run_patch(
                job["prompt_id"],
                layer_index=job["layer_index"],
                token_site=job["token_site"],
                **{job["operation"]: job["operand"]},
            )
            writer.add(job["endpoint_id"], endpoint)
        index = writer.finish()
        scores = score_saved_cells(RawReader(root, index), logical, info["model_id"])
        seal_json(root / "CELL_SCORES.json", scores)
        seal_json(root / "SUMMARY.json", api.summary.summarize_scores(scores, design))
        counts.update(
            unique_patch_count=len(jobs),
            test_executor_total=test_executor.physical_forward_count,
            total=counts["qualification"]
            + counts["fit_validation"]
            + test_executor.physical_forward_count,
        )
        if (
            sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
            > 15000000000
        ):
            raise RuntimeError("combined artifact storage limit exceeded")
        if api.capture.selection_metadata(selected) != {
            key: value for key, value in selection.items() if key != "maps_file_sha256"
        }:
            raise RuntimeError("selected map factors changed after test exposure")
        check_registration(output, source_root=source_root)
        result = {
            "status": "completed",
            "model_id": info["model_id"],
            "unique_patch_count": len(jobs),
            "interpretation": qualification["interpretation"],
            "physical_counts": counts,
        }
        seal_json(root / "RESULT.json", result)
        seal_custody(root, output, counts)
        return result
    except BaseException:
        for name, executor in (
            ("fit_executor_total", fit_executor),
            ("test_executor_total", test_executor),
        ):
            if executor is not None:
                counts[name] = executor.physical_forward_count
        seal_json(
            root / "failure.json",
            {
                "phase": "run",
                "traceback": traceback.format_exc(),
                "physical_counts": counts,
            },
        )
        raise
    finally:
        for executor in (fit_executor, test_executor):
            if executor is not None:
                executor.close()
        executor = fit_executor = test_executor = model = None
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rescore(output, model_name, *, source_root=HERE):
    if model_name not in MODELS:
        raise ValueError("model is outside the registered panel")
    output, root = Path(output), Path(output) / model_name
    check_registration(output, source_root=source_root)
    verify_file(root / "CUSTODY.json")
    custody = read_json(root / "CUSTODY.json")
    if custody["registration_sha256"] != verify_file(output / "REGISTRATION.json"):
        raise ValueError("custody belongs to a different registration")
    for name, digest in custody["artifact_sha256"].items():
        verify_file(root / name, digest)
    actual = score_saved_cells(
        RawReader(root, read_json(root / "RAW_INDEX.json")),
        read_json(root / "PLAN.json")["cells"],
        MODELS[model_name]["model_id"],
    )
    if actual != read_json(root / "CELL_SCORES.json"):
        raise ValueError("independent raw rescore differs from saved cell scores")
    if modules().summary.summarize_scores(
        actual, read_json(output / "DESIGN.json")
    ) != read_json(root / "SUMMARY.json"):
        raise ValueError("independent raw rescore differs from saved summary")
    return {
        "exact_raw_rescore_match": True,
        "exact_summary_match": True,
        "cell_count": len(actual),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run", "rescore"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(CACHE_ROOT),
        help="Persistent public HF cache; used by prepare only.",
    )
    args = parser.parse_args(argv)
    if args.command in ("run", "rescore") and args.model is None:
        parser.error("run/rescore requires --model")
    if args.command == "prepare":
        result = prepare(args.output, cache_root=args.cache_root)
    elif args.command == "check":
        check_registration(args.output)
        result = {"registration_valid": True}
    else:
        result = (run_model if args.command == "run" else rescore)(
            args.output, args.model
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
