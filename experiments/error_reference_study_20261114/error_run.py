"""Frozen development-only behavioral measurements; no causal interventions.

The previous positive-editing study stays closed. This separate runner never
forwards a confirmation question and never provisions a resource. Preparation
uses public tokenizer/configuration files only; run requires a fresh deadline.
Every output namespace is single-use, including failed attempts.
"""

import argparse
import fnmatch
import importlib
import importlib.metadata
import json
import sys
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DEPENDENCY = HERE.parent / "e30_14_work"
NEW_SOURCES = (
    "error_run.py",
    "error_protocol.py",
    "error_analysis.py",
    "test_error_run.py",
    "test_error_protocol.py",
    "test_error_analysis.py",
    "ALIAS_SELECTION.json",
)
MODELS = {
    "qwen": {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "raw_limit_bytes": 16_000_000_000,
    },
    "phi": {
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
        "raw_limit_bytes": 4_000_000_000,
    },
}
MAX_NEW_TOKENS, DEVELOPMENT_COUNT = 8, 3072
METADATA_ALLOWANCE = 256 * 1024**2
PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "numpy",
    "safetensors",
    "huggingface-hub",
)
TOKEN_PATTERNS = [
    "config.json",
    "tokenizer*",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "*.jinja",
    "generation_config.json",
]


def helpers(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    old = importlib.import_module("e30_14_run")
    for name in old.REQUIRED_SOURCES:
        module = importlib.import_module(Path(name).stem)
        if Path(module.__file__).resolve().parent != root:
            raise ValueError("frozen helper escaped its declared dependency root")
    return old


def source_manifest(old, dependency_root):
    return {
        "old": {
            name: old.sha256(Path(dependency_root) / name)
            for name in old.REQUIRED_SOURCES
        },
        "new": {name: old.sha256(HERE / name) for name in NEW_SOURCES},
    }


def validate_deadline(value, *, now=None):
    if not isinstance(value, str):
        raise TypeError("an explicit fresh UTC deadline is required")
    deadline = datetime.fromisoformat(value)
    now = datetime.now(UTC) if now is None else now
    if deadline.tzinfo is None or deadline.utcoffset() != timedelta(0):
        raise ValueError("deadline must use UTC")
    if not now < deadline <= now + timedelta(hours=6):
        raise ValueError("deadline must be in the next six hours")
    return deadline


def development_rows(tokens, split):
    if split != "development":
        raise ValueError("confirmation model calls are forbidden in this runner")
    rows = [row for row in tokens["rows"] if row["split"] == "development"]
    if (
        len(rows) != DEVELOPMENT_COUNT
        or len({row["id"] for row in rows}) != DEVELOPMENT_COUNT
    ):
        raise ValueError("every development row is required exactly once")
    return rows


def development_worlds(worlds):
    selected = [world for world in worlds if world["split"] == "development"]
    if len(selected) != 32 or len({world["world_id"] for world in selected}) != 32:
        raise ValueError("analysis requires exactly the 32 development worlds")
    return selected


def estimated_bytes(vocab_size):
    return DEVELOPMENT_COUNT * MAX_NEW_TOKENS * vocab_size * 4 + METADATA_ALLOWANCE


def token_assets(old, info, cache_root):
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            info["model_id"],
            revision=info["revision"],
            token=False,
            allow_patterns=TOKEN_PATTERNS,
            cache_dir=str(cache_root),
        )
    )
    paths = [
        path
        for path in snapshot.iterdir()
        if path.is_file()
        and any(fnmatch.fnmatch(path.name, pattern) for pattern in TOKEN_PATTERNS)
    ]
    hashes = {
        path.name: old.sha256(old.snapshot_file(path, snapshot)) for path in paths
    }
    config = old.read_json(snapshot / "config.json")
    generation = (
        old.read_json(snapshot / "generation_config.json")
        if (snapshot / "generation_config.json").is_file()
        else {}
    )
    tokenizer = old.load_tokenizer(snapshot)
    termination_ids = set()
    for value in (
        config.get("eos_token_id"),
        generation.get("eos_token_id"),
        tokenizer.eos_token_id,
    ):
        if value is not None:
            termination_ids.update(value if isinstance(value, list) else [value])
    return {
        "snapshot": str(snapshot),
        "cache_root": str(Path(cache_root).absolute()),
        "asset_sha256": hashes,
        "vocab_size": config["vocab_size"],
        "eos_token_ids": sorted(termination_ids),
    }, tokenizer


def tokenize_rows(tokenizer, rows, vocab_size):
    prepared, prefixes, geometries = [], {}, {}
    for original in rows:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": original["text"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if rendered.count(original["text"]) != 1:
            raise ValueError("chat template changed the literal question")
        encoded = tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True
        )
        ids, offsets = (
            list(encoded["input_ids"]),
            [list(pair) for pair in encoded["offset_mapping"]],
        )
        appended = list(
            tokenizer(rendered + original["expected_answer"], add_special_tokens=False)[
                "input_ids"
            ]
        )
        if (
            not ids
            or len(ids) != len(offsets)
            or appended[: len(ids)] != ids
            or len(appended) == len(ids)
        ):
            raise ValueError("answer append or prompt token geometry is invalid")
        if any(
            type(value) is not int or not 0 <= value < vocab_size for value in appended
        ):
            raise ValueError("token lies outside the output vocabulary")
        markers = {}
        for marker in ("[WORLD_END]", "[FOCUS_END]", "[ALIAS_END]"):
            if marker in original["text"]:
                start = rendered.index(marker)
                positions = [
                    i
                    for i, (left, right) in enumerate(offsets)
                    if left < start + len(marker) and right > start
                ]
                if not positions:
                    raise ValueError("registered marker has no token positions")
                markers[marker] = positions
        extra = {}
        if original.get("alias_end_span") is not None:
            shift = rendered.index(original["text"])
            start, end = [shift + point for point in original["alias_end_span"]]
            positions = [
                i
                for i, (left, right) in enumerate(offsets)
                if left < end and right > start
            ]
            if (
                not positions
                or offsets[positions[-1]][1] > shift + original["field_start"]
            ):
                raise ValueError(
                    "alias marker token includes the forbidden future field"
                )
            extra["alias_end_position"] = positions[-1]
            key = (
                original["world_id"],
                original["name_swap"],
                original["focus_condition"],
                original["query_alias_index"],
            )
            prefix = ids[: positions[-1] + 1]
            if key in prefixes and prefixes[key] != prefix:
                raise ValueError(
                    "different-attribute queries do not share an exact alias prefix"
                )
            prefixes[key] = prefix
            if original["query_alias_index"] in (0, 1):
                geometry_key = (
                    original["world_id"],
                    original["focus_condition"],
                    original["field"],
                )
                geometries.setdefault(geometry_key, set()).add(
                    (len(ids), positions[-1], len(appended) - len(ids))
                )
        prepared.append(
            {
                **original,
                **extra,
                "input_ids": ids,
                "rendered_prompt": rendered,
                "offset_mapping": offsets,
                "expected_answer_ids": appended[len(ids) :],
                "marker_positions": markers,
            }
        )
    if any(len(values) != 1 for values in geometries.values()):
        raise ValueError(
            "coreferring aliases or consistent name swaps changed token geometry"
        )
    development_rows({"rows": prepared}, "development")
    return prepared


def prepare(
    output, *, dependency_root=DEFAULT_DEPENDENCY, cache_root="/workspace/models/hub"
):
    protocol = importlib.import_module("error_protocol")
    old, output = helpers(dependency_root), Path(output)
    output.mkdir()
    try:
        study, code = protocol.build_study(), source_manifest(old, dependency_root)
        for family, files in code.items():
            target = output / "sources" / family
            target.mkdir(parents=True)
            for name in files:
                parent = Path(dependency_root) if family == "old" else HERE
                old.write_bytes(target / name, (parent / name).read_bytes())
        records = {
            "PROTOCOL.json": study["protocol"],
            "WORLDS.json": study["worlds"],
            "ROWS.json": study["rows"],
            "MODELS.json": MODELS,
            "CODE.json": code,
        }
        for name, value in records.items():
            old.seal_json(output / name, value)
        token_names = []
        for model_name, info in MODELS.items():
            tokens, tokenizer = token_assets(old, info, cache_root)
            tokens.update(
                rows=tokenize_rows(tokenizer, study["rows"], tokens["vocab_size"]),
                model_outputs_used=False,
            )
            if estimated_bytes(tokens["vocab_size"]) > info["raw_limit_bytes"]:
                raise ValueError(
                    "registered worst-case raw storage exceeds the model ceiling"
                )
            name = f"TOKENS-{model_name}.json"
            old.seal_json(output / name, tokens)
            token_names.append(name)
        if source_manifest(old, dependency_root) != code:
            raise ValueError("source changed during preparation")
        registration = {
            "dependency_root": str(Path(dependency_root).resolve()),
            "input_sha256": {
                name: old.verify_file(output / name)
                for name in (*records, *token_names)
            },
            "runtime_python": str(Path(sys.executable).absolute()),
            "python_version": sys.version,
            "runtime_versions": {
                name: importlib.metadata.version(name) for name in PACKAGES
            },
            "development_prompt_count": DEVELOPMENT_COUNT,
            "development_world_count": 32,
            "confirmation_world_count": 64,
            "confirmation_model_calls_authorized": False,
            "causal_calls_authorized": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "execution_scope": {
                "device": "cuda:0",
                "dtype": "float32",
                "attention": "eager",
                "batch_size": 1,
                "lm_head_positions": 1,
                "tf32": False,
            },
        }
        old.seal_json(output / "REGISTRATION.json", registration)
        return registration
    except BaseException:
        old.seal_json(
            output / "failure.json",
            {"phase": "prepare", "traceback": traceback.format_exc()},
        )
        raise


def check(output, dependency_root=None):
    protocol = importlib.import_module("error_protocol")
    output = Path(output)
    registration = json.loads((output / "REGISTRATION.json").read_text())
    dependency_root = dependency_root or registration["dependency_root"]
    old = helpers(dependency_root)
    old.verify_file(output / "REGISTRATION.json")
    for name, digest in registration["input_sha256"].items():
        old.verify_file(output / name, digest)
    code = old.read_json(output / "CODE.json")
    if (
        code != source_manifest(old, dependency_root)
        or old.read_json(output / "MODELS.json") != MODELS
    ):
        raise ValueError("source closure or immutable model panel changed")
    study = protocol.build_study()
    for name, value in (
        ("PROTOCOL.json", study["protocol"]),
        ("WORLDS.json", study["worlds"]),
        ("ROWS.json", study["rows"]),
    ):
        if old.read_json(output / name) != value:
            raise ValueError("registered worlds, prompts or protocol changed")
    if (
        registration["confirmation_model_calls_authorized"]
        or registration["causal_calls_authorized"]
        or registration["max_new_tokens"] != MAX_NEW_TOKENS
        or registration["development_prompt_count"] != DEVELOPMENT_COUNT
    ):
        raise ValueError("development-only execution scope changed")
    for family, files in code.items():
        for name, digest in files.items():
            old.verify_file(output / "sources" / family / name, digest)
    return old, registration


def checked_tokens(old, output, model_name):
    if model_name not in MODELS:
        raise ValueError("model is outside the immutable panel")
    tokens = old.read_json(Path(output) / f"TOKENS-{model_name}.json")
    for name, digest in tokens["asset_sha256"].items():
        old.verify_file(
            old.snapshot_file(Path(tokens["snapshot"]) / name, tokens["snapshot"]),
            digest,
        )
    return tokens


def load_model(old, info, tokens, output):
    """Compose the frozen hash guards with a new local-only model panel."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    snapshot = Path(
        snapshot_download(
            info["model_id"],
            revision=info["revision"],
            token=False,
            allow_patterns=[
                *TOKEN_PATTERNS,
                "*.safetensors",
                "*.safetensors.index.json",
            ],
            cache_dir=tokens["cache_root"],
        )
    )
    if snapshot.resolve() != Path(tokens["snapshot"]).resolve():
        raise ValueError("weight snapshot differs from the registered token snapshot")
    for name, digest in tokens["asset_sha256"].items():
        old.verify_file(old.snapshot_file(snapshot / name, snapshot), digest)
    paths = [
        path
        for path in snapshot.iterdir()
        if path.name in tokens["asset_sha256"]
        or path.name.endswith((".safetensors", ".safetensors.index.json"))
    ]
    names = {path.name for path in paths}
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("pinned snapshot has no safetensors weights")
    for path in paths:
        if path.name.endswith(".safetensors.index.json") and any(
            Path(name).name != name or name not in names
            for name in old.read_json(path)["weight_map"].values()
        ):
            raise ValueError("weight index escaped the pinned snapshot")
    fingerprints, files = {}, {}
    for path in sorted(paths):
        actual = old.snapshot_file(path, snapshot)
        stat = actual.stat()
        fingerprints[path.name] = (
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        files[path.name] = {"sha256": old.sha256(actual), "size_bytes": stat.st_size}
    old.seal_json(
        output / "MODEL_FILES.json",
        {"model_id": info["model_id"], "revision": info["revision"], "files": files},
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            local_files_only=True,
            token=False,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .to(device="cuda:0", dtype=torch.float32)
        .eval()
        .requires_grad_(False)
    )
    model.config.use_cache = False
    for path in paths:
        stat = old.snapshot_file(path, snapshot).stat()
        if fingerprints[path.name] != (
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        ):
            raise RuntimeError("pinned model file changed during loading")
    return model


def eos_ids(model, tokenizer):
    result = set()
    for source in (model.config, model.generation_config, tokenizer):
        value = source.eos_token_id
        if value is not None:
            result.update(value if isinstance(value, list) else [value])
    return result


def decode_answer(tokenizer, token_ids, eos):
    """Strip only the terminating token; other special tokens remain errors."""
    content_ids = token_ids[:-1] if token_ids and token_ids[-1] in eos else token_ids
    return tokenizer.decode(content_ids, skip_special_tokens=False)


def new_prediction(row):
    return {
        "id": row["id"],
        "expected_answer": row["expected_answer"],
        "generated_ids": [],
        "generated_text": "",
        "step_endpoint_ids": [],
        "stop_reason": None,
        "terminated": False,
        "truncated": False,
    }


def generate_row(
    model,
    tokenizer,
    row,
    prediction,
    writer,
    *,
    vocab_size,
    deadline,
    raw_limit,
    counter,
):
    import torch

    eos = eos_ids(model, tokenizer)
    for step in range(MAX_NEW_TOKENS):
        if datetime.now(UTC) >= datetime.fromisoformat(deadline):
            raise RuntimeError("execution deadline has passed")
        ids = torch.tensor(
            [row["input_ids"] + prediction["generated_ids"]],
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        with torch.inference_mode():
            logits = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            ).logits
        if (
            logits.dtype != torch.float32
            or tuple(logits.shape) != (1, 1, vocab_size)
            or not bool(torch.isfinite(logits).all())
        ):
            raise ValueError("output must be full native finite float32 vocabulary")
        native = logits[0, 0].detach().cpu()
        if counter["bytes"] + native.numel() * 4 > raw_limit:
            raise RuntimeError("raw native-logit ceiling exceeded")
        identity = prediction["id"] + f"::step{step:02d}"
        writer.add(identity, native)
        counter["bytes"] += native.numel() * 4
        token = int(native.argmax())
        prediction["generated_ids"].append(token)
        prediction["step_endpoint_ids"].append(identity)
        prediction["generated_text"] = decode_answer(
            tokenizer, prediction["generated_ids"], eos
        )
        if token in eos:
            prediction["stop_reason"] = "eos"
            prediction["terminated"] = True
            return
    prediction["stop_reason"] = "max_tokens"
    prediction["truncated"] = True


def checkpoint(old, writer, root, predictions):
    writer.flush()
    old.seal_json(
        root / f"progress-{len(predictions):06d}.json",
        {
            "completed_rows": len(predictions),
            "predictions": predictions,
            "raw_index": writer.index if hasattr(writer, "index") else None,
        },
    )
    print(
        json.dumps(
            {"completed_rows": len(predictions), "total_rows": DEVELOPMENT_COUNT}
        ),
        flush=True,
    )


def run(output, model_name, *, split, deadline, dependency_root=None):
    import torch

    validate_deadline(deadline)
    old, registration = check(output, dependency_root)
    tokens = checked_tokens(old, output, model_name)
    rows = development_rows(tokens, split)
    output, root = Path(output), Path(output) / model_name
    root.mkdir()
    writer, predictions = None, []
    try:
        old.seal_json(
            root / "EXECUTION.json",
            {
                "model": model_name,
                "split": split,
                "deadline_utc": deadline,
                "confirmation_calls_authorized": False,
                "causal_calls_authorized": False,
            },
        )
        old.seal_json(root / "RUNTIME.json", old.configure_runtime(registration))
        model = load_model(old, MODELS[model_name], tokens, root)
        if any(
            parameter.requires_grad or parameter.dtype != torch.float32
            for parameter in model.parameters()
        ) or any(module.training for module in model.modules()):
            raise ValueError("model must be frozen float32 evaluation mode")
        tokenizer, writer = old.load_tokenizer(tokens["snapshot"]), old.RawWriter(root)
        if eos_ids(model, tokenizer) != set(tokens["eos_token_ids"]):
            raise ValueError("runtime termination tokens differ from registration")
        counter = {"bytes": 0}
        for row in rows:
            prediction = new_prediction(row)
            predictions.append(prediction)
            generate_row(
                model,
                tokenizer,
                row,
                prediction,
                writer,
                vocab_size=tokens["vocab_size"],
                deadline=deadline,
                raw_limit=MODELS[model_name]["raw_limit_bytes"] - METADATA_ALLOWANCE,
                counter=counter,
            )
            if len(predictions) % 100 == 0:
                checkpoint(old, writer, root, predictions)
        writer.finish()
        old.seal_json(root / "GENERATIONS.json", predictions)
        result = importlib.import_module("error_analysis").score(
            rows, predictions, development_worlds(old.read_json(output / "WORLDS.json"))
        )
        old.seal_json(root / "RESULT.json", result)
        check(output, dependency_root)
        if (
            sum(path.stat().st_size for path in root.iterdir() if path.is_file())
            > MODELS[model_name]["raw_limit_bytes"]
        ):
            raise RuntimeError("combined behavioral artifacts exceeded storage ceiling")
        artifacts = {
            path.name: old.sha256(path)
            for path in root.iterdir()
            if path.is_file() and not path.name.endswith(".sha256")
        }
        old.seal_json(
            root / "CUSTODY.json",
            {
                "registration_sha256": old.verify_file(output / "REGISTRATION.json"),
                "artifact_sha256": artifacts,
                "physical_forward_count": sum(
                    len(item["generated_ids"]) for item in predictions
                ),
                "completed_rows": len(predictions),
            },
        )
        return result
    except BaseException:
        trace, flush_error = traceback.format_exc(), None
        if writer is not None and not writer.closed:
            try:
                writer.finish()
            except (OSError, RuntimeError, ValueError) as error:
                flush_error = {"type": type(error).__name__, "message": str(error)}
        old.seal_json(root / "PARTIAL_GENERATIONS.json", predictions)
        old.seal_json(
            root / "failure.json",
            {"phase": "run", "traceback": trace, "flush_error": flush_error},
        )
        raise


def rescore(output, model_name, *, dependency_root=None):
    old, _ = check(output, dependency_root)
    tokens = checked_tokens(old, output, model_name)
    output, root = Path(output), Path(output) / model_name
    old.verify_file(root / "CUSTODY.json")
    custody = old.read_json(root / "CUSTODY.json")
    if custody["registration_sha256"] != old.verify_file(output / "REGISTRATION.json"):
        raise ValueError("custody registration differs")
    for name, digest in custody["artifact_sha256"].items():
        old.verify_file(root / name, digest)
    predictions = old.read_json(root / "GENERATIONS.json")
    rows = development_rows(tokens, "development")
    if len(predictions) != len(rows) or {item["id"] for item in predictions} != {
        row["id"] for row in rows
    }:
        raise ValueError("saved generations do not cover development exactly once")
    reader = old.RawReader(root, old.read_json(root / "RAW_INDEX.json"))
    tokenizer, endpoints = old.load_tokenizer(tokens["snapshot"]), []
    for item in predictions:
        if (
            not 1
            <= len(item["generated_ids"])
            == len(item["step_endpoint_ids"])
            <= MAX_NEW_TOKENS
        ):
            raise ValueError("saved generation length changed")
        for step, (token, endpoint) in enumerate(
            zip(item["generated_ids"], item["step_endpoint_ids"])
        ):
            if endpoint != item["id"] + f"::step{step:02d}":
                raise ValueError("saved endpoint identity differs from its question")
            native = reader.read(endpoint)
            if (
                tuple(native.shape) != (tokens["vocab_size"],)
                or str(native.dtype) != "torch.float32"
                or int(native.argmax()) != token
            ):
                raise ValueError("saved token is not full native-logit argmax")
            endpoints.append(endpoint)
        eos_steps = [
            i
            for i, token in enumerate(item["generated_ids"])
            if token in tokens["eos_token_ids"]
        ]
        terminated = bool(eos_steps)
        if (
            (eos_steps and eos_steps != [len(item["generated_ids"]) - 1])
            or item["terminated"] != terminated
            or item["truncated"] != (not terminated)
            or item["stop_reason"] != ("eos" if terminated else "max_tokens")
            or (not terminated and len(item["generated_ids"]) != MAX_NEW_TOKENS)
        ):
            raise ValueError("saved termination/truncation contract differs")
        if (
            decode_answer(tokenizer, item["generated_ids"], tokens["eos_token_ids"])
            != item["generated_text"]
        ):
            raise ValueError("saved decoded text changed")
    if (
        len(endpoints) != len(set(endpoints))
        or set(endpoints) != set(reader.index["endpoints"])
        or custody["physical_forward_count"] != len(endpoints)
    ):
        raise ValueError("native step coverage or forward count differs")
    result = importlib.import_module("error_analysis").score(
        rows, predictions, development_worlds(old.read_json(output / "WORLDS.json"))
    )
    if result != old.read_json(root / "RESULT.json"):
        raise ValueError("independent exact behavioral rescore differs")
    return {
        "exact_match": True,
        "native_steps": len(endpoints),
        "confirmation_calls": 0,
        "causal_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run", "rescore"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dependency-root", type=Path)
    parser.add_argument("--cache-root", default="/workspace/models/hub")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--split", choices=("development",))
    parser.add_argument("--deadline")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.output,
            dependency_root=args.dependency_root or DEFAULT_DEPENDENCY,
            cache_root=args.cache_root,
        )
    elif args.command == "check":
        check(args.output, args.dependency_root)
        result = {"registration_valid": True}
    elif args.command == "run":
        result = run(
            args.output,
            args.model,
            split=args.split,
            deadline=args.deadline,
            dependency_root=args.dependency_root,
        )
    else:
        result = rescore(args.output, args.model, dependency_root=args.dependency_root)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
