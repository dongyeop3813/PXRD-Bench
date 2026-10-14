from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Optional
import torch

from src.eval_util.metric_names import metric_key, CSP, MATCH_RATE, RMSD, RATIO


@dataclass
class _BestCandidate:
    step: int
    epoch: int
    mode: str  # structure-conditioned sampling or validation
    selection_source: str  # "primary" or "fallback"
    selector_name: str
    selector_value: float
    raw_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class BestCkptState:
    best_primary: Optional[_BestCandidate] = None
    best_fallback: Optional[_BestCandidate] = None

    def state_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict):
        return cls(
            **{
                key: _BestCandidate(**value) if value is not None else None
                for key, value in state.items()
            }
        )


def _best_ckpt_csp_tag_order(best_ckpt_selector: str) -> list[str]:
    auto = ["sample_ema", "sample", "precise_ema", "precise"]
    by_selector = {
        "auto": auto,
        "legacy": auto,
        "csp_val_ratio": auto,
        "csp_precise_val_ratio": ["precise_ema", "precise"],
        "csp_sample_val_ratio": ["sample_ema", "sample"],
    }
    return by_selector.get(best_ckpt_selector, auto)


def _build_best_candidate(
    step: int,
    epoch: int,
    mode: str,
    selection_source: str,
    selector_name: str,
    selector_value: float,
    raw_metrics: dict[str, float],
) -> Optional[_BestCandidate]:
    if not math.isfinite(selector_value):
        return None
    return _BestCandidate(
        step=step,
        epoch=epoch,
        mode=mode,
        selection_source=selection_source,
        selector_name=selector_name,
        selector_value=selector_value,
        raw_metrics=dict(raw_metrics),
    )


def build_val_fallback_candidate(
    step: int, epoch: int, mode: str, val_loss: float
) -> Optional[_BestCandidate]:
    if not math.isfinite(val_loss):
        return None
    return _BestCandidate(
        step=step,
        epoch=epoch,
        mode=mode,
        selection_source="fallback",
        selector_name="val_loss",
        selector_value=val_loss,
        raw_metrics={"val_loss": val_loss},
    )


def select_primary_candidate_from_sampling(
    is_csp: bool,
    step: int,
    epoch: int,
    csp_payloads: Optional[dict[str, list[dict[str, Any]]]] = None,
    best_ckpt_selector: str = "auto",
) -> Optional[_BestCandidate]:
    if not is_csp:
        raise ValueError("Only structure-conditioned checkpoint selection is supported")
    elif best_ckpt_selector == "csp_match_rate@1":
        # Fixed-ID EMA monitor selection: RMSD is diagnostic, never a divisor.
        for tag, entries in (csp_payloads or {}).items():
            if not tag.endswith("_ema"):
                continue
            for entry in entries:
                if entry.get("csp_source_label") != "val" or not entry.get(
                    "ordered_ids_sha256"
                ):
                    continue
                if (
                    entry.get("matcher_profile") != "strict"
                    or entry.get("match_gate") != "matcher_only"
                ):
                    raise ValueError(
                        "K1 checkpoint selection requires strict matcher-only metrics"
                    )
                value = entry["csp_match_rate@1"]
                return _build_best_candidate(
                    step,
                    epoch,
                    "csp",
                    "primary",
                    f"{tag}/csp_match_rate@1",
                    value,
                    {f"{tag}/csp_match_rate@1": value},
                )
        return None
    else:
        # CSP: use val match_rate / mean_rms as selector.
        csp_payloads = csp_payloads or {}
        for tag in _best_ckpt_csp_tag_order(best_ckpt_selector):
            entries = csp_payloads.get(tag)
            if entries is None:
                continue
            val_entry = None
            for entry in entries:
                if entry.get("csp_source_label") == "val":
                    val_entry = entry
                    break
            if val_entry is None:
                continue
            match_rate = val_entry.get("match_rate")
            mean_rms = val_entry.get("mean_rms")
            if match_rate is None or mean_rms is None:
                continue
            if not math.isfinite(match_rate) or not math.isfinite(mean_rms):
                continue
            if mean_rms == 0.0:
                continue
            ratio = match_rate / mean_rms
            if not math.isfinite(ratio):
                continue
            raw = {
                f"{tag}/{metric_key(CSP, MATCH_RATE, 1)}": match_rate,
                f"{tag}/{metric_key(CSP, RMSD, 1)}": mean_rms,
                f"{tag}/{metric_key(CSP, RATIO)}": ratio,
            }
            return _build_best_candidate(
                step=step,
                epoch=epoch,
                mode="csp",
                selection_source="primary",
                selector_name=f"{tag}/{metric_key(CSP, RATIO)}",
                selector_value=ratio,
                raw_metrics=raw,
            )
        return None


def _save_best_checkpoint(
    ckpt_dir: Path, candidate: _BestCandidate, build_ckpt_fn
) -> None:
    """Persist selected weights and metadata using the existing temporary paths."""
    ckpt = build_ckpt_fn(candidate.step)
    torch.save(ckpt, ckpt_dir / ".best.tmp.pt")
    (ckpt_dir / ".best.tmp.pt").replace(ckpt_dir / "best.pt")
    meta = {
        "step": candidate.step,
        "epoch": candidate.epoch,
        "mode": candidate.mode,
        "selection_source": candidate.selection_source,
        "selector_name": candidate.selector_name,
        "selector_value": candidate.selector_value,
        "raw_metrics": candidate.raw_metrics,
        "updated_at_step": candidate.step,
    }
    (ckpt_dir / ".best_meta.tmp.json").write_text(json.dumps(meta, indent=2))
    (ckpt_dir / ".best_meta.tmp.json").replace(ckpt_dir / "best_meta.json")


def maybe_update_best_ckpt(
    state: BestCkptState,
    candidate: _BestCandidate,
    maximize: bool,
    ckpt_dir: Path,
    build_ckpt_fn,
    enabled: bool,
) -> bool:
    """Update best checkpoint state and save to disk when improved.

    Returns True if the on-disk best.pt was updated.
    """
    if not enabled:
        return False

    is_primary = candidate.selection_source != "fallback"
    tol = 1e-12

    if is_primary:
        prev = state.best_primary
        if prev is not None:
            if maximize:
                improved = candidate.selector_value > prev.selector_value + tol
            else:
                improved = candidate.selector_value < prev.selector_value - tol
        else:
            improved = True
        if improved:
            state.best_primary = candidate
            _save_best_checkpoint(ckpt_dir, candidate, build_ckpt_fn)
            return True
        return False
    else:
        # Fallback: always track the best fallback value
        prev_fb = state.best_fallback
        improved = prev_fb is None or (
            (not maximize and candidate.selector_value < prev_fb.selector_value - tol)
            or (maximize and candidate.selector_value > prev_fb.selector_value + tol)
        )
        if improved:
            state.best_fallback = candidate

        # Save the first/improved fallback only while no primary exists.
        if improved and state.best_primary is None:
            _save_best_checkpoint(ckpt_dir, candidate, build_ckpt_fn)
            return True
        return False


def resolve_post_training_eval_ckpt(ckpt_dir: Path) -> Path | None:
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        return best_path
    final_path = ckpt_dir / "final.pt"
    if final_path.exists():
        return final_path
    return None


def select_monitor_candidate(step, epoch, payloads):
    """Select the fixed IID monitor's Match@1, including zero/perfect RMS cases."""
    for tag in sorted(payloads, key=lambda tag: not tag.endswith("_ema")):
        for entry in payloads[tag]:
            if entry.get("csp_source_label") == "val":
                value = float(entry["match_rate"])
                key = f"{tag}/csp_match_rate@1"
                return _build_best_candidate(
                    step, epoch, "csp", "primary", key, value, {key: value}
                )
    return None
