"""Label-blind finite decomposition of one already selected attention head.

This is a fixed two-by-two profile/pattern comparison at recipient record total,
plus total-record, nonrecord-residual and whole-head diagnostics. The random
control matches ONLY the whole-head delta norm. Nothing here accepts answer
labels or uses correctness to choose a direction.
"""

import hashlib
import math

import torch

CONDITIONS = (
    "self",
    "oldDcolorMass",
    "sameFieldMass",
    "sameFieldWithinOnly",
    "sameFieldProfileWithin",
    "sameFieldRecordContribution",
    "sameFieldNonrecordResidual",
    "sameFieldFullHead",
    "normMatchedRandomFullHead",
)
TOLERANCE = 2e-5
SEED = "fixed-head-nine-cell-decomposition-20261115-v1"


def _checked(operand, regions):
    a, values, head = (operand[key] for key in ("weights", "values", "head_output"))
    if any(
        not isinstance(v, torch.Tensor)
        or v.dtype != torch.float32
        or v.device.type != "cpu"
        or not bool(torch.isfinite(v).all())
        for v in (a, values, head)
    ):
        raise ValueError("weights, V and actual head must be finite native CPU float32")
    if (
        a.ndim != 1
        or values.ndim != 2
        or values.shape[0] != len(a)
        or tuple(head.shape) != (values.shape[1],)
        or bool((a < 0).any())
        or abs(float(a.double().sum()) - 1) > 1e-6
    ):
        raise ValueError("native head, value or probability geometry is invalid")
    positions = [p for obj in "ABCD" for p in regions[obj]]
    if (
        set(regions) != set("ABCD")
        or any(not regions[obj] for obj in "ABCD")
        or len(set(positions)) != len(positions)
        or any(type(p) is not int or not 0 <= p < len(a) - 1 for p in positions)
    ):
        raise ValueError("printed records must have disjoint nonempty pre-query masks")
    masses = torch.stack([a[regions[obj]].double().sum() for obj in "ABCD"])
    total = float(masses.sum())
    if total < 1e-4:
        raise ValueError("record total lacks support; no epsilon rescue")
    reconstruction = a @ values
    error = float((reconstruction.double() - head.double()).abs().max())
    if (
        not bool(torch.isfinite(reconstruction).all())
        or not math.isfinite(error)
        or error > TOLERANCE
    ):
        raise ValueError(
            "actual head differs from the CPU full contraction by more than2e-5"
        )
    return a, values, head, masses, total, masses / total, error


def decompose(
    recipient, donor, donor_color, regions, *, case_id, field, old_color_profile=None
):
    """Compute nine frozen per-field deltas, with every proof tensor retained."""
    if torch.get_num_threads() != 1:
        raise ValueError("frozen CPU contractions require one thread")
    ar, vr, hr, mr, tr, pr, er = _checked(recipient, regions)
    ad, vd, hd, md, td, pd, ed = _checked(donor, regions)
    ac, vc, _, mc, tc, pc, ec = _checked(donor_color, regions)
    if (
        ar.shape != ad.shape
        or ar.shape != ac.shape
        or vr.shape != vd.shape
        or vr.shape != vc.shape
        or bool((mr < 1e-8).any())
        or bool((md < 1e-8).any())
    ):
        raise ValueError(
            "donor AND recipient per-record pattern support/geometry is required"
        )
    positions = sorted(p for values in regions.values() for p in values)
    record_bytes = vr[positions].contiguous().numpy().tobytes()
    if any(
        v[positions].contiguous().numpy().tobytes() != record_bytes for v in (vd, vc)
    ):
        raise ValueError("original donor/recipient record V must be bytewise identical")
    if old_color_profile is not None:
        old = torch.as_tensor(old_color_profile, dtype=torch.float64)
        if tuple(old.shape) != (4,) or not torch.allclose(old, pc, rtol=0, atol=1e-14):
            raise ValueError("old color profile differs from its exact replayed source")
        pc = old

    def factor(condition):
        coefficients = torch.zeros(len(ar), dtype=torch.float64)
        for index, obj in enumerate("ABCD"):
            region = regions[obj]
            if condition in ("oldDcolorMass", "sameFieldMass"):
                profile = pc if condition == "oldDcolorMass" else pd
                coefficients[region] = (
                    (profile[index] - pr[index])
                    * tr
                    * (ar[region].double() / mr[index])
                )
            elif condition == "sameFieldWithinOnly":
                coefficients[region] = (mr[index] / md[index]) * ad[
                    region
                ].double() - ar[region].double()
            else:
                coefficients[region] = (tr / td) * ad[region].double() - ar[
                    region
                ].double()
        dw = coefficients.float()
        changed = ar + dw
        masses = torch.stack([changed[regions[obj]].double().sum() for obj in "ABCD"])
        profile_error = float(
            (
                masses / masses.sum()
                - (
                    pr
                    if condition == "sameFieldWithinOnly"
                    else pc
                    if condition == "oldDcolorMass"
                    else pd
                )
            )
            .abs()
            .max()
        )
        if (
            bool((changed < 0).any())
            or abs(float(masses.sum()) - tr) > 1e-7
            or profile_error > 1e-6
        ):
            raise ValueError(
                "factor kernel violated recipient total/profile numerical bounds"
            )
        return {
            "delta": dw @ vr,
            "delta_weights": dw,
            "target_weights": changed,
            "record_total_error": abs(float(masses.sum()) - tr),
            "profile_max_abs_error": profile_error,
        }

    cr, cd = ar[positions] @ vr[positions], ad[positions] @ vd[positions]
    nr, nd = hr - cr, hd - cd
    full, record, nonrecord = hd - hr, cd - cr, nd - nr
    diff_weights = torch.zeros_like(ar)
    diff_weights[positions] = (ad[positions].double() - ar[positions].double()).float()
    equivalent_record = diff_weights @ vr
    equivalence = float((record.double() - equivalent_record.double()).abs().max())
    reconstruction = float(((record + nonrecord).double() - full.double()).abs().max())
    full_target = float(((hr + full).double() - hd.double()).abs().max())
    if (
        not all(math.isfinite(v) for v in (equivalence, reconstruction, full_target))
        or max(equivalence, reconstruction, full_target) > TOLERANCE
    ):
        raise ValueError("record/residual/full-head reconstruction exceeds2e-5")
    seed = int.from_bytes(
        hashlib.sha256(f"{SEED}:{case_id}:{field}:28:0".encode()).digest()[:8], "big"
    ) % (2**63)
    norm = float(torch.linalg.vector_norm(full.double()))
    random = torch.zeros_like(full)
    if norm:
        direction = torch.randn(
            len(full),
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float64,
        )
        random = (direction / torch.linalg.vector_norm(direction) * norm).float()
    norm_error = (
        abs(float(torch.linalg.vector_norm(random.double())) - norm) / norm
        if norm
        else 0.0
    )
    if norm_error > 1e-5:
        raise ValueError("random whole-head norm mismatch exceeds1e-5")
    conditions = {"self": {"delta": torch.zeros_like(hr)}}
    for name in CONDITIONS[1:5]:
        conditions[name] = factor(name)
    conditions.update(
        sameFieldRecordContribution={
            "delta": record,
            "equivalent_delta_weights": diff_weights,
            "equivalent_delta": equivalent_record,
        },
        sameFieldNonrecordResidual={"delta": nonrecord},
        sameFieldFullHead={"delta": full},
        normMatchedRandomFullHead={
            "delta": random,
            "seed": seed,
            "matched_condition": "sameFieldFullHead",
            "relative_norm_error": norm_error,
        },
    )
    for value in conditions.values():
        delta = value["delta"]
        if not bool(torch.isfinite(delta).all()):
            raise ValueError("a decomposition delta is not finite")
        value["delta_norm"] = float(torch.linalg.vector_norm(delta.double()))
    return {
        "conditions": conditions,
        "recipient_record_contribution": cr,
        "donor_record_contribution": cd,
        "recipient_nonrecord_residual": nr,
        "donor_nonrecord_residual": nd,
        "recipient_masses": mr,
        "donor_masses": md,
        "color_masses": mc,
        "recipient_total": tr,
        "donor_total": td,
        "color_total": tc,
        "recipient_profile": pr,
        "donor_profile": pd,
        "color_profile": pc,
        "audit": {
            "recipient_full_contraction_max_abs": er,
            "donor_full_contraction_max_abs": ed,
            "color_full_contraction_max_abs": ec,
            "record_delta_equivalence_max_abs": equivalence,
            "full_reconstruction_max_abs": reconstruction,
            "full_target_max_abs": full_target,
            "record_V_sha256": hashlib.sha256(record_bytes).hexdigest(),
        },
        "numeric_scope": "CPUFP64 mass coefficients→CPUFP32 delta@recipientV; C separately CPUFP32 contracted; N=actualh−C; fullhead/random not matched to other delta norms",
    }
