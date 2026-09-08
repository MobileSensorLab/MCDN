"""Common utilities for the modeling-chapter visualization scripts.

Establishes a single publication-style baseline (seaborn ``paper`` context,
sans-serif default, perceptually-ordered ordinal damage palette) and a
``save_figure`` helper that writes both an SVG (primary, used by HTML render)
and a PDF sidecar (used by the LaTeX render) to ``outputs/images/``, mirroring
both to ``doc/images/`` for Quarto chapter resolution. Chapter ``![]()`` refs
omit the file extension and rely on Quarto's ``default-image-extension``
mechanism to dispatch to the right format per render target. This avoids
the SVG -> PDF conversion step in the LaTeX pipeline (which depends on a
non-Python ``rsvg-convert`` binary that is not available on Windows under
the project's uv-only env management).

Result lineage (revision 2, work-plan D-12/D-19): every metric-driven figure
reads the DGX 10-seed runs under ``outputs/ablation_dgx/<variant>/<split>/``
and the ensemble artifacts under ``outputs/ablation_dgx/_ensembles/
<variant>__<split>{.json,_probs.pt}``. The full configuration is the
``all_features`` variant; the four reported evaluation columns are listed in
``REPORTED_COLUMNS`` with their D-20 display labels (the DROIDs default split
first, then the three LOEO holdouts). ``load_ensemble_summary`` /
``ensemble_rule_metrics`` / ``per_seed_rule_metrics`` expose the JSON.

Also provides ``load_reference_val_chip`` for data-dependent figures. The
helper rebuilds one column's validation dataset once per process (through the
same fold logic the ensemble tooling uses) and indexes it against the cached
ensemble probability tensor so chip selection is deterministic, replayable,
and class-conditionable. The default holdout is Mayfield Tornado, the
distant-OOD column, so the exemplars are drawn from the most challenging
operating context rather than the easiest.

Intended consumption pattern from each ``fig_<name>.py`` script::

    from scripts.visualization._common import (
        ORDINAL_CLASS_NAMES,
        ORDINAL_CLASS_PALETTE,
        WIDTH_2COL,
        save_figure,
        setup_publication_style,
    )

    def main() -> None:
        setup_publication_style()
        fig, ax = plt.subplots(figsize=(WIDTH_2COL, 3.2))
        ...
        save_figure(fig=fig, name="my_figure")
"""
from __future__ import annotations

import functools
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Final

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.figure import Figure

# Repository-rooted paths. ``__file__`` -> scripts/visualization/_common.py;
# parents[2] is the repo root.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_OUTPUTS_DIR: Final[Path] = _REPO_ROOT / "outputs" / "images"
_DOC_MIRROR_DIR: Final[Path] = _REPO_ROOT / "doc" / "images"

# Default holdout that the data-dependent figures align to. The Mayfield
# Tornado distant-OOD split grounds the visual exemplars in the most
# challenging operating context. Other holdouts can be selected via the
# ``holdout`` argument on ``load_reference_val_chip``; the mask-alignment
# exemplar uses Hurricane_Michael for its left panel and Mayfield_Tornado for
# its right panel.
_DEFAULT_HOLDOUT: Final[str] = "Mayfield_Tornado"

# Revision-2 result lineage: DGX 10-seed runs and their ensemble artifacts.
_LINEAGE_DIR: Final[Path] = _REPO_ROOT / "outputs" / "ablation_dgx"
_ENSEMBLE_DIR: Final[Path] = _LINEAGE_DIR / "_ensembles"
_DATA_DIR: Final[Path] = _REPO_ROOT / "data"

# The full MCDN configuration (mask channel + mask-weighted pooling + typology FiLM).
FULL_VARIANT: Final[str] = "all_features"

# Split directory of the dataset's published train/test partition (the
# composite multi-event holdout) and the four reported columns in D-20 order
# with their display labels. The spatial fold is appendix-only and not listed.
DEFAULT_SPLIT_DIR: Final[str] = "Hurricane_Idalia+Hurricane_Michael+Mayfield_Tornado+Mussett_Bayou_Fire"
REPORTED_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    (DEFAULT_SPLIT_DIR, "DROIDs default"),
    ("Hurricane_Michael", "LOEO Michael"),
    ("Mayfield_Tornado", "LOEO Mayfield"),
    ("Hurricane_Ida", "LOEO Ida")
)
COLUMN_LABELS: Final[dict[str, str]] = dict(REPORTED_COLUMNS)


def _holdout_fold_dir(holdout: str, variant: str = FULL_VARIANT) -> Path:
    """Return the seed_00 fold directory for the given holdout name."""

    return _LINEAGE_DIR / variant / holdout / "seed_00"


def _holdout_probs_path(holdout: str, variant: str = FULL_VARIANT) -> Path:
    """Return the cached ensemble probability tensor for the given holdout name."""

    return _ENSEMBLE_DIR / f"{variant}__{holdout}_probs.pt"


def ensemble_summary_path(split: str, variant: str = FULL_VARIANT) -> Path:
    """Return the ensemble summary JSON path for one ``(variant, split)`` pair."""

    return _ENSEMBLE_DIR / f"{variant}__{split}.json"


@functools.lru_cache(maxsize=64)
def load_ensemble_summary(split: str, variant: str = FULL_VARIANT) -> dict[str, Any]:
    """Load the ensemble summary JSON written by ``scripts/run_per_arm_ensembles.py``.

    Args:
        split: Split directory name (one of the ``REPORTED_COLUMNS`` keys or
            ``"Spatial_Block_East"``).
        variant: Ablation variant directory name under ``outputs/ablation_dgx/``.

    Returns:
        The parsed JSON payload (``variants[0]`` holds per-seed replay metrics
        and ``cross_variant.equal_seed_metrics`` the ensemble metrics per rule).

    Raises:
        FileNotFoundError: If the summary has not been rendered for this pair.
    """

    path = ensemble_summary_path(split, variant)
    if not path.exists():
        raise FileNotFoundError(f"Ensemble summary not found: {path}. Run scripts/run_per_arm_ensembles.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def ensemble_rule_metrics(split: str, rule: str = "argmax", variant: str = FULL_VARIANT) -> dict[str, Any]:
    """Return the 10-seed ensemble metrics block for one split under one decoding rule.

    The block carries ``qwk``, ``accuracy``, ``macro_f1``, ``macro_recall``,
    ``macro_precision``, ``per_class_f1`` / ``per_class_recall`` /
    ``per_class_precision`` (dicts keyed by ``ORDINAL_CLASS_NAMES``) and
    ``confusion_matrix`` (``[true][pred]`` nested lists).
    """

    return load_ensemble_summary(split, variant)["cross_variant"]["equal_seed_metrics"][rule]


def per_seed_rule_metrics(split: str, rule: str = "argmax", variant: str = FULL_VARIANT) -> list[dict[str, Any]]:
    """Return the per-seed replayed metrics blocks (same schema as ``ensemble_rule_metrics``) in seed order."""

    return [entry["replay_metrics"][rule] for entry in load_ensemble_summary(split, variant)["variants"][0]["per_seed"]]


if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Page widths (inches) calibrated to standard journal column widths.
WIDTH_1COL: Final[float] = 3.5
WIDTH_2COL: Final[float] = 7.0

# IEEE asks for figure text at 8 pt when the figure is printed at its final size, with
# consistent sizes across figures. Every figure script draws labels, ticks, titles and
# legends at FIG_FONT_PT; FIG_ANNOT_PT is the floor for in-figure annotations (cell text,
# bar values) where 8 pt does not fit. Figures whose canvas is wider than the column they
# are placed in must scale these up by the reduction factor (see fig_mcdn_block).
FIG_FONT_PT: Final[float] = 8.0
FIG_ANNOT_PT: Final[float] = 7.0

# Ordinal damage-scale display names. Mirrors ``src.model.trainer.ORDINAL_CLASS_DISPLAY_NAMES``
# and ``src.data.dataset.ORDINAL_MAP`` so the visualization layer reports the same labels
# the trainer and metrics pipeline use.
ORDINAL_CLASS_NAMES: Final[tuple[str, ...]] = ("No Damage", "Minor", "Major", "Destroyed")

# FEMA-aligned qualitative palette for the four damage classes. The hex values are
# drawn from ColorBrewer Set1 (Cynthia Brewer's perceptually-tuned qualitative scheme)
# and map to the FEMA Damage Assessment Guide's color convention: green=safe,
# orange=minor, red=major, purple=destroyed. ColorBrewer Set1 was selected over
# the literal FEMA-guide hexes for its peer-review-defensible perceptual distinctness
# and reasonable (though not perfect) colorblind behavior. Per-figure scripts must
# pair these colors with redundant non-color encoding (text labels, hatching, or
# marker shapes) so figures remain parseable for color-vision-deficient readers and
# under black-and-white print reproduction.
ORDINAL_CLASS_PALETTE: Final[tuple[str, ...]] = (
    "#4DAF4A",  # No Damage  (Set1 green)
    "#FF7F00",  # Minor      (Set1 orange)
    "#E41A1C",  # Major      (Set1 red)
    "#984EA3",  # Destroyed  (Set1 purple)
)

_STYLE_APPLIED: bool = False


def setup_publication_style() -> None:
    """Apply the project-wide publication-style baseline. Idempotent.

    Invoked at the start of every ``fig_<name>.py::main()``. Calling more than
    once in the same process is a no-op (the apply flag is module-scoped).

    Style commitments:
        - Seaborn ``paper`` context with ``style="white"`` (clean axes, no grid).
        - Sans-serif font stack led by Arial / Helvetica (IEEE-accepted figure
          fonts) with DejaVu Sans as the always-available fallback;
          ``svg.fonttype="none"`` preserves text as text in exported SVGs so
          downstream consumers (Quarto, browsers, LaTeX) can re-style if needed,
          and ``pdf.fonttype=42`` embeds the TrueType font in the PDF sidecar.
        - All text at ``FIG_FONT_PT`` (8 pt, the IEEE figure-text size at final
          column width); in-figure annotations may drop to ``FIG_ANNOT_PT`` where
          8 pt does not fit.
        - Default categorical palette ``colorblind`` for non-ordinal usage; the
          ordinal damage palette is exposed separately as ``ORDINAL_CLASS_PALETTE``.
        - Top and right axes spines hidden; tight layout reserved per-figure.
    """

    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    sns.set_theme(style="white", context="paper", font_scale=1.0, palette="colorblind")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        # Route mathtext through the same sans face so Greek letters and deltas in
        # axis labels match the surrounding text; glyphs Arial lacks fall back to DejaVu.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": FIG_FONT_PT,
        "axes.labelsize": FIG_FONT_PT,
        "axes.titlesize": FIG_FONT_PT,
        "xtick.labelsize": FIG_FONT_PT,
        "ytick.labelsize": FIG_FONT_PT,
        "legend.fontsize": FIG_FONT_PT,
        "figure.titlesize": FIG_FONT_PT,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    _STYLE_APPLIED = True


def _validate_name(name: str) -> str:
    """Validate a figure name and return the bare stem (no extension)."""

    if not name:
        raise ValueError("name must be non-empty.")
    if "/" in name or "\\" in name:
        raise ValueError(f"name must not contain a path separator (got {name!r}).")
    if name.endswith(".svg"):
        return name[:-4]
    return name


def save_figure(fig: Figure, name: str, *, close: bool = True) -> Path:
    """Save ``fig`` as both SVG and PDF under ``outputs/images/`` and ``doc/images/``.

    Each output directory is created on first call. SVG is the primary
    artifact (consumed by Quarto's HTML render and by external review); PDF is
    a sidecar consumed by the LaTeX render so we don't depend on a system-level
    ``rsvg-convert`` binary at build time. Chapter ``![]()`` refs omit the file
    extension and rely on Quarto's ``default-image-extension`` mechanism to
    dispatch (``svg`` for HTML, ``pdf`` for LaTeX).

    Both formats are produced from the same matplotlib ``Figure`` object, so
    the visual content is bit-identical modulo the format-specific text-glyph
    handling matplotlib applies (``svg.fonttype="none"`` keeps SVG text as
    text; PDF embeds Type-3 outlines from the same font stack).

    On-figure title and caption text is **not** the right venue for a publication
    figure - the chapter's ``fig-cap`` directive is authoritative, and matplotlib's
    synthesized-italic font rendering produces awkward letter-spacing on long
    captions anyway. The companion ``save_caption`` helper writes the suggested
    chapter-caption Markdown alongside the figure so consumers see both at the
    same path stem.

    Args:
        fig: Matplotlib ``Figure`` to write.
        name: Output stem. ``.svg`` extension is stripped if present. Must be
            non-empty and free of path separators.
        close: When True (default), the figure is closed via ``plt.close(fig)``
            after writing to release matplotlib backend resources. Set False if
            the caller intends to display or further manipulate the figure.

    Returns:
        Path to the primary SVG under ``outputs/images/``. The PDF sidecar
        and the ``doc/images/`` mirrors are written but not returned; consumers
        should not depend on those paths directly.

    Raises:
        ValueError: If ``name`` is empty or contains a path separator.
    """

    stem = _validate_name(name)
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _DOC_MIRROR_DIR.mkdir(parents=True, exist_ok=True)

    canonical_svg = _OUTPUTS_DIR / f"{stem}.svg"
    fig.savefig(
        canonical_svg,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=False,
    )
    shutil.copy2(canonical_svg, _DOC_MIRROR_DIR / f"{stem}.svg")

    canonical_pdf = _OUTPUTS_DIR / f"{stem}.pdf"
    fig.savefig(
        canonical_pdf,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=False,
    )
    shutil.copy2(canonical_pdf, _DOC_MIRROR_DIR / f"{stem}.pdf")

    if close:
        plt.close(fig)

    return canonical_svg


def save_caption(name: str, *, title: str, body: str) -> Path:
    """Save a sibling Markdown caption alongside an existing figure.

    The convention is one ``.caption.md`` file per figure stem, written to both
    ``outputs/images/<name>.caption.md`` (primary) and ``doc/images/<name>.caption.md``
    (mirror). Chapter authors can either inline the caption text into a
    ``![caption](images/<name>.svg)`` directive or read it via Quarto include
    shortcodes; either way, the suggested caption travels with the figure
    artifact and survives palette / data tweaks that don't touch the editorial
    framing.

    The Markdown body is wrapped under a level-1 heading carrying the suggested
    figure title, so the file renders as a self-contained caption document when
    previewed standalone.

    Args:
        name: Output stem (matching the corresponding ``save_figure`` call).
            ``.caption.md`` is appended automatically.
        title: One-line suggested figure title (becomes the level-1 Markdown
            heading and the natural Quarto ``fig-cap`` source). Inline math may
            use Markdown's ``$ ... $`` fences.
        body: Multi-paragraph editorial framing - what the figure shows, what
            the relevant constants are, and any caveats. Markdown formatting is
            permitted (bold, italics, math, references).

    Returns:
        Path to the primary caption Markdown under ``outputs/images/``.

    Raises:
        ValueError: If ``name`` is invalid or ``title`` / ``body`` are empty.
    """

    if not title:
        raise ValueError("title must be non-empty.")
    if not body:
        raise ValueError("body must be non-empty.")

    stem = _validate_name(name)
    canonical_path = _OUTPUTS_DIR / f"{stem}.caption.md"
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text(f"# {title.strip()}\n\n{body.strip()}\n", encoding="utf-8")

    mirror_path = _DOC_MIRROR_DIR / f"{stem}.caption.md"
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(canonical_path, mirror_path)

    return canonical_path


@functools.lru_cache(maxsize=6)
def _build_reference_val_dataset(holdout: str) -> tuple[Any, np.ndarray, np.ndarray]:
    """Construct one holdout's val dataset and load its cached ensemble probs.

    Cached at module level via ``lru_cache`` so the (slow) manifest scan and
    polygon-load step happens once per holdout per process even when multiple
    chips are requested; ``maxsize=6`` covers the four reported columns, the
    spatial fold and a spare. Delegates fold reconstruction to
    ``src.postproc.ensemble.get_fold_dataframes_from_config`` so dataset
    indices line up with the cached probability rows (including the
    multi-event default split).

    Args:
        holdout: Split directory name under ``outputs/ablation_dgx/all_features/``
            (e.g. ``"Mayfield_Tornado"``, ``"Hurricane_Ida"``, ``DEFAULT_SPLIT_DIR``).

    Returns:
        Tuple of (dataset, targets [N], ensemble_probs [N, K]).
    """

    # Local imports keep the visualization-style baseline import-light when
    # only ``setup_publication_style`` / ``save_figure`` are needed.
    import torch
    import yaml

    from src.data.dataset import CRASARUnitemporalDataset
    from src.postproc.ensemble import get_fold_dataframes_from_config

    config_path = _holdout_fold_dir(holdout) / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Resolved config snapshot not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # The DGX snapshots record the cluster data path; rebuild the fold against the local staging.
    _train_df, val_df, _holdout_label = get_fold_dataframes_from_config(cfg, data_dir_override=str(_DATA_DIR))

    dataset = CRASARUnitemporalDataset(
        manifest=val_df,
        chip_size=cfg["data"]["chip_size"],
        transform=None,
        is_train=False,
        cache_validation_tensors=False,
        mask_dilation_px=cfg["ablation"].get("mask_dilation_px", 0),
    )

    probs_path = _holdout_probs_path(holdout)
    if not probs_path.exists():
        raise FileNotFoundError(f"Ensemble probs cache not found: {probs_path}. Run scripts/run_per_arm_ensembles.py first.")
    cache = torch.load(probs_path, map_location="cpu", weights_only=False)
    targets = cache["targets"].numpy().astype(np.int64)
    ensemble_probs = cache["variants"][0]["ensemble_probs"].numpy().astype(np.float32)

    if len(dataset) != targets.shape[0]:
        raise RuntimeError(
            f"Dataset size ({len(dataset)}) does not match cached probs row count "
            f"({targets.shape[0]}); the fold's manifest may have drifted since the ensemble was rendered."
        )

    return dataset, targets, ensemble_probs


def load_reference_val_pool(holdout: str = _DEFAULT_HOLDOUT) -> dict[str, Any]:
    """Load the full val pool for one holdout.

    Returns the val dataset, targets, per-seed individual probs, and the
    softmax-mean ensemble probs - everything the decoding-rule figure needs to
    render per-seed bar charts plus an ensemble bar chart for one specific
    manifest index.

    Args:
        holdout: Split directory name under ``outputs/ablation_dgx/all_features/``.
            Defaults to the module-level ``_DEFAULT_HOLDOUT``
            (``"Mayfield_Tornado"``).

    Returns:
        Dict with:
          - ``dataset``: the ``CRASARUnitemporalDataset`` keyed by manifest
            index aligned to the cached probability tensors.
          - ``targets``: ``ndarray`` int64 ``[N]`` ground-truth class labels.
          - ``individual_probs``: ``ndarray`` float32 ``[n_seeds, N, K]``
            softmax probabilities per seed.
          - ``ensemble_probs``: ``ndarray`` float32 ``[N, K]`` softmax-mean
            across the seed pool.
          - ``seeds``: list of seed-name strings (e.g. ``["seed_00", ...]``).
          - ``holdout``: the holdout name (echoed back).
    """

    import torch

    dataset, targets, ensemble_probs = _build_reference_val_dataset(holdout)
    cache = torch.load(_holdout_probs_path(holdout), map_location="cpu", weights_only=False)
    variant = cache["variants"][0]
    individual_probs = variant["individual_probs"].numpy().astype(np.float32)
    seeds = [f"seed_{int(seed):02d}" for seed in variant["seeds"]]

    return {
        "dataset": dataset,
        "targets": targets,
        "individual_probs": individual_probs,
        "ensemble_probs": ensemble_probs,
        "seeds": seeds,
        "holdout": holdout,
    }


def load_reference_val_chip(
    *,
    class_name: str | None = None,
    rng_seed: int = 0,
    require_correct: bool = True,
    holdout: str = _DEFAULT_HOLDOUT,
) -> dict[str, Any]:
    """Load a representative chip from one full-configuration val set.

    The dataset is the same one the ``all_features`` arm validated on at the
    requested holdout; chip indexing is aligned to the cached ensemble
    probabilities so any chip we render also has its ensemble softmax
    available for downstream display.

    Default holdout is ``Mayfield_Tornado`` (the distant-OOD column); the
    mask-alignment exemplar overrides to ``Hurricane_Michael`` for its left panel.

    Args:
        class_name: If provided, restrict candidates to the given true class.
            Must be one of ``ORDINAL_CLASS_NAMES``. ``None`` allows any class.
        rng_seed: Deterministic ranking position within the candidate pool.
            ``0`` returns the most-confident candidate; higher values walk
            down the confidence ranking.
        require_correct: When True (default), restrict to candidates where
            the ensemble's argmax matches the true class. False allows
            inspection of misclassifications.
        holdout: Split directory name under ``outputs/ablation_dgx/all_features/``.
            Defaults to ``Mayfield_Tornado``.

    Returns:
        Dict with:
          - ``rgb``: ``ndarray`` uint8 ``[H, W, 3]``.
          - ``mask``: ``ndarray`` uint8 ``[H, W]`` in ``{0, 1}``.
          - ``true_class_idx``: int in ``[0, K-1]``.
          - ``true_class_name``: matching ``ORDINAL_CLASS_NAMES`` entry.
          - ``ensemble_softmax``: ``ndarray`` float32 ``[K]``.
          - ``ensemble_argmax_idx``: int in ``[0, K-1]``.
          - ``manifest_idx``: int row index into the val dataset.
          - ``confidence_rank``: int (0 = most confident in the candidate
            pool, ``rng_seed`` returns this rank).
          - ``holdout``: the holdout name the chip was drawn from.

    Raises:
        ValueError: If ``class_name`` is not in ``ORDINAL_CLASS_NAMES``.
        RuntimeError: If no candidates match the filter (most likely with
            ``class_name`` and ``require_correct=True`` on a class with no
            confident correct predictions in this holdout).
    """

    dataset, targets, ensemble_probs = _build_reference_val_dataset(holdout)

    if class_name is not None and class_name not in ORDINAL_CLASS_NAMES:
        raise ValueError(
            f"class_name must be one of {ORDINAL_CLASS_NAMES}, got {class_name!r}."
        )

    ensemble_argmax = ensemble_probs.argmax(axis=1)

    candidate_mask = np.ones(targets.shape[0], dtype=bool)
    if class_name is not None:
        target_idx = ORDINAL_CLASS_NAMES.index(class_name)
        candidate_mask &= (targets == target_idx)
    if require_correct:
        candidate_mask &= (ensemble_argmax == targets)

    candidates = np.where(candidate_mask)[0]
    if candidates.size == 0:
        raise RuntimeError(
            f"No candidate chips match (class_name={class_name!r}, "
            f"require_correct={require_correct}). Adjust filters and retry."
        )

    # Rank candidates by ensemble confidence in the true class. Higher
    # confidence = lower rank index, which makes ``rng_seed=0`` the
    # exemplar-of-record for the requested class.
    confidence = ensemble_probs[candidates, targets[candidates]]
    sorted_local = np.argsort(-confidence)
    rank = min(int(rng_seed), sorted_local.size - 1)
    chosen_idx = int(candidates[sorted_local[rank]])

    sample = dataset[chosen_idx]
    image_uint8 = sample["image"].numpy()  # [4, H, W]
    rgb = image_uint8[:3].transpose(1, 2, 0).copy()
    mask = image_uint8[3].copy()

    return {
        "rgb": rgb,
        "mask": mask,
        "true_class_idx": int(targets[chosen_idx]),
        "true_class_name": ORDINAL_CLASS_NAMES[int(targets[chosen_idx])],
        "ensemble_softmax": ensemble_probs[chosen_idx].copy(),
        "ensemble_argmax_idx": int(ensemble_argmax[chosen_idx]),
        "manifest_idx": chosen_idx,
        "confidence_rank": rank,
        "holdout": holdout,
    }
