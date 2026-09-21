"""Frozen Q-only quality contract shared by A3 candidate finalization/training."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from a3_io import A3Error, sha256_file

EXPECTED_WEIGHTS = {"F": 0.40, "P": 0.35, "D": 0.25}
EXPECTED_RULE = "Q-only"
EXPECTED_THRESHOLD = 0.55
QUALITY_REPORT_SCHEMA = "quality-report-v1"


def load_quality_contract(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise A3Error(f"Missing frozen quality config: {path}")
    try:
        import yaml
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (ImportError, OSError, UnicodeError, ValueError) as exc:
        raise A3Error(f"Invalid frozen quality config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A3Error(f"Frozen quality config must be an object: {path}")
    if value.get("version") != "v1" or value.get("rule") != EXPECTED_RULE:
        raise A3Error("Formal A3 requires quality_v1 Q-only")
    threshold = value.get("threshold", {}).get("Q")
    if threshold != EXPECTED_THRESHOLD:
        raise A3Error(f"Q threshold must be {EXPECTED_THRESHOLD}, got {threshold!r}")
    components = value.get("components")
    if not isinstance(components, dict):
        raise A3Error("quality config lacks components")
    actual_weights = {
        name: components.get(name, {}).get("weight") for name in EXPECTED_WEIGHTS
    }
    if actual_weights != EXPECTED_WEIGHTS:
        raise A3Error(
            f"Q weights must be {EXPECTED_WEIGHTS}, got {actual_weights}"
        )
    return {
        "version": "v1",
        "rule": EXPECTED_RULE,
        "threshold_Q": EXPECTED_THRESHOLD,
        "weights": dict(EXPECTED_WEIGHTS),
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def load_quality_report(path: str | Path,
                        contract: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise A3Error(f"Missing quality report: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise A3Error(f"Invalid quality report {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise A3Error("Quality report lacks a results list")
    report_schema = payload.get("schema_version")
    if report_schema != QUALITY_REPORT_SCHEMA:
        raise A3Error(
            f"Formal candidates require {QUALITY_REPORT_SCHEMA}; got {report_schema!r}. "
            "Unversioned historical reports require an explicit audited conversion."
        )
    declared_contract = payload.get("quality_contract")
    if not isinstance(declared_contract, dict):
        raise A3Error("Quality report lacks the frozen quality_contract binding")
    expected_contract = {
        "sha256": contract["sha256"], "version": contract["version"],
        "rule": contract["rule"],
    }
    for key, expected in expected_contract.items():
        if declared_contract.get(key) != expected:
            raise A3Error(
                f"Quality report contract {key} mismatch: "
                f"{declared_contract.get(key)!r} != {expected!r}"
            )
    report_weights = payload.get("weights")
    expected_report_weights = {
        "fidelity": contract["weights"]["F"],
        "physics": contract["weights"]["P"],
        "diversity": contract["weights"]["D"],
    }
    if report_weights != expected_report_weights:
        raise A3Error(
            f"Quality report weights mismatch: {report_weights!r}"
        )
    if payload.get("thresholds", {}).get("total") != contract["threshold_Q"]:
        raise A3Error("Quality report total threshold is not frozen Q=0.55")
    declared = payload.get("filter_rule")
    expected_filter = {
        "name": EXPECTED_RULE, "threshold_Q": EXPECTED_THRESHOLD,
        "component_hard_gates": [],
    }
    if declared != expected_filter:
        raise A3Error(f"Quality report declares a non-Q-only filter: {declared!r}")

    declared_inventory = payload.get("candidate_inventory")
    if not isinstance(declared_inventory, dict):
        raise A3Error("Quality report lacks candidate_inventory binding")
    if declared_inventory.get("schema_version") != "a3-candidate-inventory-v1":
        raise A3Error("Quality report candidate_inventory schema mismatch")
    inventory_sha = declared_inventory.get("sha256")
    inventory_count = declared_inventory.get("row_count")
    inventory_method = declared_inventory.get("method")
    if (not isinstance(inventory_sha, str) or len(inventory_sha) != 64 or
            any(char not in "0123456789abcdef" for char in inventory_sha)):
        raise A3Error("Quality report candidate_inventory SHA-256 is invalid")
    if (isinstance(inventory_count, bool) or not isinstance(inventory_count, int) or
            inventory_count <= 0):
        raise A3Error("Quality report candidate_inventory row_count is invalid")
    if inventory_method not in {"TRAD", "LOADSIM"}:
        raise A3Error("Quality report candidate_inventory method is invalid")

    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(payload["results"]):
        if not isinstance(raw, dict):
            raise A3Error(f"Quality result {index} is not an object")
        sample_id = raw.get("variant_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in by_id:
            raise A3Error(f"Invalid/duplicate quality variant_id: {sample_id!r}")
        detail = raw.get("fidelity_detail")
        if not isinstance(detail, dict) or detail.get("estimated") is not False:
            raise A3Error(
                f"Formal A3 requires explicitly measured Fidelity for {sample_id}"
            )
        hashes = {
            "source_ply_sha256": raw.get("source_ply_sha256"),
            "candidate_ply_sha256": raw.get("candidate_ply_sha256"),
        }
        for label, digest in hashes.items():
            if (not isinstance(digest, str) or len(digest) != 64 or
                    any(char not in "0123456789abcdef" for char in digest)):
                raise A3Error(f"Invalid/missing {label} for {sample_id}")
        values = {
            "F": raw.get("fidelity_score"),
            "P": raw.get("physics_score"),
            "D": raw.get("diversity_contribution"),
            "Q": raw.get("total_score"),
        }
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(float(value)) for value in values.values()):
            raise A3Error(f"Non-finite quality score for {sample_id}")
        scores = {key: float(value) for key, value in values.items()}
        if any(not 0.0 <= score <= 1.0 for score in scores.values()):
            raise A3Error(f"Quality score outside [0,1] for {sample_id}: {scores}")
        expected_q = sum(contract["weights"][key] * scores[key]
                         for key in ("F", "P", "D"))
        if not math.isclose(scores["Q"], expected_q, rel_tol=0.0, abs_tol=1e-9):
            raise A3Error(f"Q formula mismatch for {sample_id}")
        expected_passed = scores["Q"] >= contract["threshold_Q"]
        if not isinstance(raw.get("passed"), bool) or raw["passed"] != expected_passed:
            raise A3Error(f"Q-only pass decision mismatch for {sample_id}")
        by_id[sample_id] = {
            **scores, "passed": expected_passed,
            "source_ply_sha256": hashes["source_ply_sha256"],
            "candidate_ply_sha256": hashes["candidate_ply_sha256"],
        }
    if not by_id:
        raise A3Error("Quality report contains no valid results")
    if len(by_id) != inventory_count:
        raise A3Error(
            "Quality report result count does not match candidate_inventory row_count")
    provenance = {
        "quality_rule": contract["rule"],
        "quality_threshold_Q": contract["threshold_Q"],
        "quality_weights": contract["weights"],
        "quality_config_sha256": contract["sha256"],
        "quality_report_sha256": sha256_file(path),
        "candidate_inventory_schema": declared_inventory["schema_version"],
        "candidate_inventory_sha256": inventory_sha,
        "candidate_inventory_count": inventory_count,
        "candidate_inventory_method": inventory_method,
    }
    return by_id, provenance
