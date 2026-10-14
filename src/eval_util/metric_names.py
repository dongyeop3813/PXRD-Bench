"""Canonical evaluation-metric naming — single source of truth.

Two layers, intentionally kept separate:

1. **Outward names** (wandb sections / report labels / saved metric dicts) follow
   a conditioning prefix (``csp_`` /
   ``pxrd_``), a leaf (``match_rate`` / ``rmsd`` / ``ratio``), and an optional
   ``@k`` suffix. **Build these only with** :func:`metric_key` so the grammar
   lives in one place and a typo (``match@1`` vs ``match_rate@1``) is impossible.
   The reference-set qualifier that prefixes them (``{tag}/...``) comes from
   :func:`refset_tag`.

2. **RecEval internal keys** (:data:`REC_MATCH_RATE`, :data:`REC_RMSD`) are the
   raw reconstruction diagnostic keys returned by
   ``RecEvalBatch.get_match_rate_and_rms*``. These are a *stable internal
   contract* consumed in several places (live training path and checkpoint files); their string values are
   deliberately **not** changed. Reference them by symbol in new code.
"""

from __future__ import annotations

# --- conditioning prefixes --------------------------------------------------
CSP = "csp"
PXRD = "pxrd"

# --- outward metric leaves ----------------------------------------------------
MATCH_RATE = "match_rate"
RMSD = "rmsd"
RATIO = "ratio"

# --- RecEval internal dict keys (NOT outward names; stable, do not rename) -----
REC_MATCH_RATE = "match_rate"
REC_RMSD = "rms_dist"


def metric_key(cond, leaf, k=None):
    """Build a canonical outward metric name (the part after the ``{tag}/``).

    ``cond``    conditioning prefix — :data:`CSP` / :data:`PXRD`.
    ``leaf``    metric leaf — :data:`MATCH_RATE` / :data:`RMSD` / :data:`RATIO`.
    ``k``       optional candidate count appended as ``@{k}`` (omit for ``ratio``).

    Examples::

        metric_key(CSP, MATCH_RATE, 1)               -> "csp_match_rate@1"
        metric_key(CSP, RMSD, 20)                    -> "csp_rmsd@20"
        metric_key(PXRD, MATCH_RATE, 5)              -> "pxrd_match_rate@5"
        metric_key(CSP, RATIO)                       -> "csp_ratio"
    """
    body = f"{cond}_{leaf}"
    return body if k is None else f"{body}@{k}"


def refset_tag(dataset_name, variant=None, split="test"):
    """Reference-set tag used as the wandb section / report qualifier.

    Examples::

        refset_tag("real_pxrd_all", "all", "test")  -> "rpxrd_all"
        refset_tag("real_pxrd_all", "atom25", "test") -> "rpxrd_a25"
        refset_tag("mp20", None, "val")              -> "mp20_val"
    """
    name = str(dataset_name or "").lower()
    if name.startswith("real_pxrd") or name.startswith("rpxrd"):
        v = str(variant or "all").lower()
        tag = "rpxrd_a25" if v in ("atom25", "a25", "atom_nmax25") else "rpxrd_all"
    elif name.startswith("mp20"):
        tag = "mp20"
    else:
        tag = name or "eval"
    s = str(split or "test").lower()
    return tag if s == "test" else f"{tag}_{s}"


def monitoring_section(dataset_name, variant=None, split="val", *, weights="ema"):
    """Return a section that always identifies dataset, split, and weights."""
    split_name = str(split or "unknown").strip().lower()
    reference = refset_tag(dataset_name, variant, split_name)
    if split_name == "test":
        reference = f"{reference}_test"
    if weights is None:
        return reference
    weights_name = str(weights).strip().lower()
    if weights_name not in {"ema", "raw"}:
        raise ValueError("monitoring weights must be ema, raw, or None")
    return f"{reference}_{weights_name}"
