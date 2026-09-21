"""All-valid activation trust-root and optimizer entry gate.

Chain of custody (§5 of the all-valid engineering directive):

    bundle receipt SHA  ─┐
    bundle manifest SHA ─┤→ catalog binds both (per method), bundle-external
                          │
    catalog SHA ─────────┴→ activation binds the catalog SHA + approved
                            implementation closure

The trust root (``_TRUST_ROOT``) is a LOCALLY-FIXED, reviewed constant: the
expected activation SHA, the expected catalog SHA and the approved
issuer/verifier implementation closure.  Ordinary CLI invocations can neither
self-sign nor substitute these values.  ``verify_activation`` fails closed
whenever any link in the chain is missing or drifts.

IMPORTANT: activation only proves that the all-valid assets are *allowed to be
consumed*.  It is NOT training authorization; the two-factor formal-run gate
(``--authorize-formal-run`` + ``LSDA_FORMAL_RUN_AUTHORIZED=YES``) is still
required and is enforced separately by the entrypoints.

The real activation/catalog are produced on the server only after both method
bundles, the field contract, the config/catalog and the test evidence are
finalized and the approved implementation SHAs are pinned.  Until then the
trust root is a fail-closed sentinel (``None``), so every gate denies.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

_WINDOWS_ABS = re.compile(r"^[A-Za-z]:")

from a3_io import (
    ALL_VALID_ACTIVATION_SCHEMA,
    ALL_VALID_CATALOG_SCHEMA,
    ALL_VALID_ELIGIBILITY_RULE,
    ALL_VALID_RECEIPT_SCHEMA,
)

CATALOG_RELPATH = "configs/a3_all_valid_contract_v1.json"
ACTIVATION_RELPATH = "activation/a3_all_valid_activation_v1.json"

# Issuer/verifier implementation whose bytes the catalog+activation pin.  These
# are the modules that MINT and VERIFY the all-valid chain (distinct from the
# frozen historical producers).
_APPROVED_IMPL_CLOSURE = ("a3_activation.py", "a3_all_valid.py", "a3_io.py")

ALL_VALID_METHODS = ("trad", "loadsim")


class ActivationError(RuntimeError):
    """Raised when the all-valid activation chain is missing or fails to verify."""


# The locally-fixed trust root lives in a SEPARATE module (``a3_trust_root``)
# that is intentionally NOT part of ``_APPROVED_IMPL_CLOSURE``.  This breaks the
# self-reference cycle: pinning the trust root edits only ``a3_trust_root.py``,
# so the bytes of a3_activation.py / a3_all_valid.py / a3_io.py — and therefore
# the ``impl_closure`` the activation binds — stay unchanged.  ``None`` means no
# reviewed activation has been pinned yet and every gate MUST fail closed.
from a3_trust_root import TRUST_ROOT as _TRUST_ROOT


def _trust_root() -> dict[str, Any] | None:
    """Return the pinned trust root, or ``None`` if none has been reviewed/fixed.

    Reads the value from the closure-external ``a3_trust_root`` module.  Kept as
    a function so tests can patch a self-consistent fixture in place of the
    (intentionally empty) production sentinel; the CLI has no path to this.
    """
    return _TRUST_ROOT


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value)


def _constrained_relpath(value: Any, base_dir: Path, label: str) -> Path:
    """Resolve a catalog-declared path that MUST be a normalized relative POSIX
    path contained under ``base_dir``.  Rejects absolute paths, backslashes,
    ``..``/`.`/empty segments and any root escape (fail-closed)."""
    if not isinstance(value, str) or not value.strip():
        raise ActivationError(f"{label} must be a non-empty relative path")
    if "\\" in value or PurePosixPath(value).is_absolute() or _WINDOWS_ABS.match(value):
        raise ActivationError(f"{label} must be a relative POSIX path: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ActivationError(f"{label} is not a normalized relative path: {value!r}")
    target = base_dir.joinpath(*PurePosixPath(value).parts)
    try:
        target.resolve().relative_to(base_dir.resolve())
    except ValueError as exc:
        raise ActivationError(f"{label} escapes the catalog base dir: {value!r}") from exc
    return target


def _no_symlink_ancestor(path: Path, base_dir: Path, label: str) -> None:
    """Fail if ``path`` or any ancestor up to ``base_dir`` is a symlink."""
    base_dir = base_dir.resolve()
    current = path
    while True:
        if current.is_symlink():
            raise ActivationError(f"{label} traverses a symlink: {current}")
        if current.resolve() == base_dir:
            return
        parent = current.parent
        if parent == current:  # reached filesystem root without hitting base_dir
            raise ActivationError(f"{label} is outside the catalog base dir: {path}")
        current = parent


def _required_leaf_sha(path: Path, base_dir: Path, label: str) -> str:
    """Return the SHA-256 of a REQUIRED catalog leaf, computed from a stable file
    descriptor.  The leaf must exist, be a regular non-symlink single-link file
    with no symlink ancestor; anything else fails closed."""
    _no_symlink_ancestor(path, base_dir, label)
    if path.is_symlink():
        raise ActivationError(f"{label} must not be a symlink: {path}")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError as exc:
        raise ActivationError(f"required catalog leaf missing for {label}: {path}") from exc
    except OSError as exc:
        raise ActivationError(f"cannot open required leaf for {label}: {exc}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ActivationError(f"{label} must be a regular file: {path}")
        if st.st_nlink != 1:
            raise ActivationError(
                f"{label} must be a single-link file (nlink={st.st_nlink}): {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(fd)

# __ACTIVATION_HELPERS__
def _current_impl_closure(base_dir: Path) -> dict[str, str]:
    """Actual SHAs of the current issuer/verifier implementation modules."""
    here = Path(__file__).resolve().parent
    closure: dict[str, str] = {}
    for name in _APPROVED_IMPL_CLOSURE:
        path = here / name
        if not path.is_file():
            raise ActivationError(f"approved implementation module missing: {name}")
        closure[name] = _sha256_file(path)
    return closure


def _load_json_strict(path: Path, label: str) -> tuple[dict[str, Any], str, bytes]:
    if not path.is_file():
        raise ActivationError(f"{label} file is missing: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ActivationError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ActivationError(f"{label} must be a JSON object")
    return value, _sha256_bytes(raw), raw


def _verify_catalog(catalog: dict[str, Any], base_dir: Path) -> None:
    """Validate the catalog structure and its per-method receipt/manifest bindings."""
    if catalog.get("schema_version") != ALL_VALID_CATALOG_SCHEMA:
        raise ActivationError("catalog schema is not all-valid v1")
    if catalog.get("eligibility_rule") != ALL_VALID_ELIGIBILITY_RULE:
        raise ActivationError("catalog eligibility rule mismatch")
    methods = catalog.get("methods")
    if not isinstance(methods, dict) or sorted(methods) != sorted(ALL_VALID_METHODS):
        raise ActivationError(
            f"catalog must bind exactly methods {ALL_VALID_METHODS}")
    for method in ALL_VALID_METHODS:
        entry = methods[method]
        if not isinstance(entry, dict):
            raise ActivationError(f"catalog method entry must be an object: {method}")
        required = {"bundle_dir", "receipt_sha256", "manifest_sha256"}
        if set(entry) != required:
            raise ActivationError(
                f"catalog method {method} fields mismatch; expected {sorted(required)}")
        if not _hex64(entry["receipt_sha256"]) or not _hex64(entry["manifest_sha256"]):
            raise ActivationError(f"catalog method {method} SHA fields must be 64-hex")
        # Bundle-external binding: the catalog names the bundle dir and the
        # SEPARATE final receipt + manifest SHAs (never the catalog's own SHA,
        # avoiding a hash cycle).  Both leaves are REQUIRED: they must exist as
        # regular, single-link, symlink-free files under a constrained relative
        # bundle dir, and their stable-fd SHA must match the catalog binding.
        bundle_dir = _constrained_relpath(
            entry["bundle_dir"], base_dir, f"catalog method {method} bundle_dir")
        _no_symlink_ancestor(bundle_dir, base_dir,
                             f"catalog method {method} bundle_dir")
        receipt = bundle_dir / "all_valid_receipt.json"
        manifest = bundle_dir / "candidate_manifest.csv"
        receipt_sha = _required_leaf_sha(
            receipt, base_dir, f"catalog method {method} receipt")
        if receipt_sha != entry["receipt_sha256"]:
            raise ActivationError(f"catalog method {method} receipt SHA drift")
        manifest_sha = _required_leaf_sha(
            manifest, base_dir, f"catalog method {method} manifest")
        if manifest_sha != entry["manifest_sha256"]:
            raise ActivationError(f"catalog method {method} manifest SHA drift")

def verify_activation(base_dir: Path) -> dict[str, Any]:
    """Fail-closed verification of the full activation chain.

    Returns the parsed activation object on success; raises ``ActivationError``
    on any missing/invalid/mismatched link.  This is the single authoritative
    gate that every optimizer entrypoint calls BEFORE model construction, and
    which ``--skip-hash-verification`` cannot reach or disable.
    """
    base_dir = Path(base_dir).resolve()
    root = _trust_root()
    if root is None:
        raise ActivationError(
            "all-valid activation is not authorized: no reviewed trust root is "
            "pinned in a3_activation._TRUST_ROOT (fail-closed)")
    for key in ("activation_sha256", "catalog_sha256", "impl_closure"):
        if key not in root:
            raise ActivationError(f"trust root is malformed: missing {key}")

    activation_path = base_dir.joinpath(*PurePosixPath(ACTIVATION_RELPATH).parts)
    activation, activation_sha, _ = _load_json_strict(activation_path, "activation")
    # 1) The activation FILE bytes must equal the reviewer-pinned SHA.  The CLI
    #    cannot forge this: changing the file changes its SHA and fails here.
    if activation_sha != root["activation_sha256"]:
        raise ActivationError(
            "activation file SHA does not match the pinned trust root "
            f"(actual={activation_sha}, pinned={root['activation_sha256']})")
    if activation.get("schema_version") != ALL_VALID_ACTIVATION_SCHEMA:
        raise ActivationError("activation schema is not all-valid v1")
    if activation.get("eligibility_rule") != ALL_VALID_ELIGIBILITY_RULE:
        raise ActivationError("activation eligibility rule mismatch")
    if activation.get("receipt_schema") != ALL_VALID_RECEIPT_SCHEMA:
        raise ActivationError("activation receipt schema mismatch")

    # 2) The activation must declare the pinned catalog SHA, and the on-disk
    #    catalog bytes must hash to it.
    declared_catalog_sha = activation.get("catalog_sha256")
    if not _hex64(declared_catalog_sha):
        raise ActivationError("activation catalog_sha256 must be 64-hex")
    if declared_catalog_sha != root["catalog_sha256"]:
        raise ActivationError(
            "activation declares a catalog SHA that differs from the pinned root")
    catalog_path = base_dir.joinpath(*PurePosixPath(CATALOG_RELPATH).parts)
    catalog, catalog_sha, _ = _load_json_strict(catalog_path, "catalog")
    if catalog_sha != declared_catalog_sha:
        raise ActivationError(
            "on-disk catalog SHA does not match the activation binding "
            f"(actual={catalog_sha}, declared={declared_catalog_sha})")

    # 3) The approved implementation closure must match current bytes exactly,
    #    both as pinned in the trust root and as declared in the activation.
    current = _current_impl_closure(base_dir)
    if root["impl_closure"] != current:
        raise ActivationError(
            "issuer/verifier implementation bytes drifted from the approved "
            "closure pinned in the trust root")
    if activation.get("impl_closure") != current:
        raise ActivationError(
            "activation-declared implementation closure differs from current bytes")

    # 4) Structural + binding checks on the catalog itself.
    _verify_catalog(catalog, base_dir)
    return activation

def is_all_valid_config(config: dict[str, Any]) -> bool:
    """True when a resolved run config belongs to the all-valid namespace and
    therefore requires the activation gate.  Q-v1 configs return False and use
    their own existing two-factor authorization instead."""
    return config.get("eligibility_rule") == ALL_VALID_ELIGIBILITY_RULE


def build_catalog(base_dir: Path, methods: dict[str, str]) -> dict[str, Any]:
    """Construct the catalog object binding each method bundle's receipt+manifest
    SHAs (bundle-external).  ``methods`` maps method→bundle_dir (relative POSIX).

    This is used by the reviewed server flow and by tests; it never mints a
    trust root.  Callers persist the returned object and pin its SHA by review.
    """
    if sorted(methods) != sorted(ALL_VALID_METHODS):
        raise ActivationError(f"catalog build requires methods {ALL_VALID_METHODS}")
    base_dir = Path(base_dir).resolve()
    entries: dict[str, Any] = {}
    for method in ALL_VALID_METHODS:
        rel = methods[method]
        bundle_dir = base_dir.joinpath(*PurePosixPath(rel).parts)
        receipt = bundle_dir / "all_valid_receipt.json"
        manifest = bundle_dir / "candidate_manifest.csv"
        for path, label in ((receipt, "receipt"), (manifest, "manifest")):
            if not path.is_file():
                raise ActivationError(f"cannot build catalog: {method} {label} missing")
        entries[method] = {
            "bundle_dir": rel,
            "receipt_sha256": _sha256_file(receipt),
            "manifest_sha256": _sha256_file(manifest),
        }
    return {
        "schema_version": ALL_VALID_CATALOG_SCHEMA,
        "eligibility_rule": ALL_VALID_ELIGIBILITY_RULE,
        "methods": entries,
    }


def build_activation(base_dir: Path, catalog_sha256: str) -> dict[str, Any]:
    """Construct the activation object binding the catalog SHA + current approved
    implementation closure.  Persisting + pinning its SHA is a reviewed step."""
    if not _hex64(catalog_sha256):
        raise ActivationError("catalog_sha256 must be 64-hex")
    return {
        "schema_version": ALL_VALID_ACTIVATION_SCHEMA,
        "eligibility_rule": ALL_VALID_ELIGIBILITY_RULE,
        "receipt_schema": ALL_VALID_RECEIPT_SCHEMA,
        "catalog_sha256": catalog_sha256,
        "impl_closure": _current_impl_closure(base_dir),
        "note": ("activation proves assets are consumable; it is NOT training "
                 "authorization (still requires --authorize-formal-run + "
                 "LSDA_FORMAL_RUN_AUTHORIZED=YES)"),
    }

