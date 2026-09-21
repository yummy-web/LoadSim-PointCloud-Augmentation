"""All-valid activation trust root (isolated, closure-external).

This module exists SOLELY to hold the reviewer-pinned trust-root constant for
the all-valid activation chain.  It is deliberately kept OUT of
``a3_activation._APPROVED_IMPL_CLOSURE`` so that pinning the trust root (editing
``TRUST_ROOT`` below) does not change the bytes of any file whose SHA the
activation binds.  That breaks the self-reference cycle that would otherwise
arise from ``a3_activation.py`` both defining the trust root and being a member
of its own approved implementation closure.

``TRUST_ROOT is None`` is the production sentinel: no reviewed activation has
been pinned, so every activation gate MUST fail closed.  When the server
bundles are finalized and verified, a reviewer replaces ``None`` with the
concrete dict (never via CLI):

    TRUST_ROOT = {
        "activation_sha256": "<64-hex>",  # SHA of the activation JSON file bytes
        "catalog_sha256": "<64-hex>",     # SHA the activation must declare
        "impl_closure": {                 # approved issuer/verifier module bytes
            "a3_activation.py": "<64-hex>",
            "a3_all_valid.py": "<64-hex>",
            "a3_io.py": "<64-hex>",
        },
    }

Editing this value is the ONLY step needed to fix the trust root; the approved
implementation modules (a3_activation.py / a3_all_valid.py / a3_io.py) keep
their bytes, so their pinned ``impl_closure`` stays self-consistent.
"""
from __future__ import annotations

from typing import Any

# Pinned 2026-09-13 after independent review: the reviewer recomputed the
# activation bytes from the reviewed a3_activation.py (9630072c) + catalog SHA
# 0bdcfe5f and reproduced activation_sha256 1a81c72e byte-for-byte; the catalog
# binds the two stage-5-verified all-valid bundles (trad/loadsim, each total=288,
# fresh_deep_replay_v1); impl_closure equals the reviewed local module bytes.
TRUST_ROOT: dict[str, Any] | None = {
    "activation_sha256": "1a81c72e5604bf610f6e7634a3152fa8290ab9fbb5cd31003f8d3a43cfaaaa01",
    "catalog_sha256": "0bdcfe5fda37fbbdff002d9e8034c43772a3fb34d96e9a0f032e04ab726df9ca",
    "impl_closure": {
        "a3_activation.py": "9630072c33fbac1e06f35d51fdfdcd1f9a1b29d4ff6ea72284a95a4f2b0650d9",
        "a3_all_valid.py": "63c450d459c0913cf163bae4eefa35cf89ca24eede28243f08d8d58a4ae6507b",
        "a3_io.py": "c68f9d25af48f55923de2199f89c4e4ddfc585c33ad658f2dc5a6379f995831f",
    },
}
