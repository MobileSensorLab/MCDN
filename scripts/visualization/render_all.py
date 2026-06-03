"""One-command regeneration of every modeling-chapter figure.

Imports each ``fig_*`` module and invokes its ``main()`` in sequence. Each
module is independently invokable (``uv run python -m scripts.visualization.fig_*``)
and writes its outputs to ``outputs/images/`` plus the ``doc/images/`` mirror;
this driver just chains them so palette tweaks in ``_common.py`` or chip
selection changes can propagate to all figures with one invocation.

Usage::

    uv run python -m scripts.visualization.render_all
"""
from __future__ import annotations

import importlib
import time
from typing import Final

# Module names in the order they should run. The data-dependent figures
# (F1 enhancement, F5, F6, F7, F8) all share the same Mayfield val dataset
# loaded by ``load_canonical_val_chip``; that dataset is module-cached via
# ``functools.lru_cache`` in ``_common.py`` so the first such figure to run
# pays the ~5 s manifest-scan cost and the rest reuse the cache.
_FIGURE_MODULES: Final[tuple[str, ...]] = (
    # Chapter 4 / paper-Methodology figures.
    "scripts.visualization.fig_label_smoothing",        # F3 (synthetic)
    "scripts.visualization.fig_convnext_hierarchy",     # F1 (data-dependent: chip at Input)
    "scripts.visualization.fig_convnext_block",         # F2 (synthetic)
    "scripts.visualization.fig_mcdn_block",             # F-MCDN (synthetic; paper Methodology)
    "scripts.visualization.fig_pooling_decomposition",  # F5 (data-dependent: chip thumbnails)
    "scripts.visualization.fig_chip_exemplars",         # F6 (data-dependent: per-class chips)
    "scripts.visualization.fig_d4_group",               # F7 (data-dependent: D4 of base chip)
    "scripts.visualization.fig_augmentation_gallery",   # F8 (data-dependent: augmented chips)
    # Chapter 5 (results) figures - all read from JSON artifacts under
    # outputs/ablation/<arm>/<split>/ rather than the canonical val dataset,
    # so they run independently and do not benefit from the cached dataset.
    "scripts.visualization.fig_confusion_triptych",     # R1 (JSON-driven)
    "scripts.visualization.fig_precision_triptych",     # R1b precision view (JSON-driven)
    "scripts.visualization.fig_ablation_deltas",        # R2 (JSON-driven)
    "scripts.visualization.fig_per_class_funnel",       # R3 (JSON-driven)
    "scripts.visualization.fig_per_seed_dispersion",    # R4 (JSON-driven)
    # Chapter 6 (discussion) figures. D1 and D2 are data-dependent on the
    # canonical val pool (extending the chapter-4 chip-loading helpers); D4
    # depends on the embedding cache produced once by
    # scripts/visualization/extract_canonical_embeddings.py. D3 is an in-chapter mermaid
    # flowchart and not a driver target.
    "scripts.visualization.fig_mask_alignment_exemplar",  # D1 (data-dependent)
    "scripts.visualization.fig_decoding_rule_example",    # D2 (data-dependent + per-seed probs)
    "scripts.visualization.fig_embedding_tsne",           # D4 (cache-driven)
)


def main() -> None:
    """Render every modeling-chapter figure in order, with per-figure timing."""

    overall_start = time.perf_counter()
    print(f"Rendering {len(_FIGURE_MODULES)} figures (8 chapter-4 + 5 chapter-5 + 3 chapter-6)...\n")

    for module_name in _FIGURE_MODULES:
        short_name = module_name.rsplit(".", 1)[-1]
        figure_start = time.perf_counter()
        print(f"  -> {short_name}", flush=True)
        module = importlib.import_module(module_name)
        module.main()
        elapsed = time.perf_counter() - figure_start
        print(f"     ({elapsed:.1f}s)")

    total_elapsed = time.perf_counter() - overall_start
    print(f"\nAll figures rendered in {total_elapsed:.1f}s.")


if __name__ == "__main__":
    main()
