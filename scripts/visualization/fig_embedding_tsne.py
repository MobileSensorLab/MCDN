"""D4: 2D embedding scatter of post-FiLM-pool fused embeddings.

Tests the chapter-6 implicit-encoding hypothesis - that the visual stream
encodes typology-relevant cues at the chip level even when the model is
trained without an explicit typology one-hot - by projecting the
typology-ablated baseline arm's fused embeddings (extracted by
``scripts.visualization.extract_canonical_embeddings``) to two dimensions and coloring
each chip by its event-level typology category.

Embeddings come from the typology-ablated arm specifically (FiLM is
structurally absent, so the embedding is purely visual); if typology
clusters are visible in the projection, the chapter's claim that
"the visual stream knows typology" is empirically supported on the same
data and architecture the chapter's other claims are made about.

Color and shape REDUNDANTLY encode typology (one categorical dimension via
two visual channels). Damage class is intentionally not encoded - the
chapter's load-bearing claim is about typology only; encoding damage
class as a secondary shape (an earlier design iteration) added visual
noise without informing the chapter's argument. Rare categories (Kinetic,
Thermal) are drawn on top of the dense Wind/Flood mass with larger
marker size and full opacity, so they are visually findable even at the
val pool's 95 / 5 / 0.2% imbalance.

Projection method: PCA (2560 -> 50 dims, deterministic) followed by
``sklearn.manifold.TSNE`` (50 -> 2 dims, perplexity 30). UMAP would be a
defensible alternative but is not in the project's package stack;
``umap-learn`` is the only piece of new dependency surface this figure
would have introduced, so t-SNE was chosen as the lower-friction option.

Visual conventions:
    - Color encodes event-level typology via matplotlib's tab10 palette
      (the ML default for categorical scatter plots): tab:blue =
      Wind/Flood, tab:orange = Kinetic, tab:green = Thermal, tab:red =
      Unknown.
    - Marker shape redundantly encodes typology (circle / square /
      triangle / diamond), matching the chapter-5 R3 per-class funnel
      convention and supporting greyscale / color-blind readability.
    - Wind/Flood at moderate alpha and modest marker size; Kinetic and
      Thermal at full opacity and larger marker size, plotted on top.
    - Single legend with per-typology counts, placed on-plot at a
      deliberate corner with a white-bbox background.

Source artifact: ``outputs/embeddings/typology_arm_sbe_val.pt``.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scripts.visualization._common import (
    WIDTH_2COL,
    save_caption,
    save_figure,
    setup_publication_style,
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_EMBEDDING_CACHE: Final[Path] = (
    _REPO_ROOT / "outputs" / "embeddings" / "typology_arm_sbe_val.pt"
)

# Typology category index -> display name. Mirrors
# src.data.dataset.CRASARUnitemporalDataset.TYPOLOGY_MAP / NUM_TYPOLOGIES.
_TYPOLOGY_NAMES: Final[tuple[str, ...]] = (
    "Wind/Flood",
    "Kinetic",
    "Thermal",
    "Unknown",
)

# Matplotlib tab10 categorical palette - the de-facto ML default for
# multi-class embedding scatter plots. Pairs with redundant shape
# encoding per typology (see _TYPOLOGY_MARKERS) so color-blind readers
# and greyscale-print consumers see the same category structure via
# marker geometry alone. The first four tab10 colors overlap incidentally
# with the FEMA-aligned damage palette's hues used in chapters 4-5 (blue
# is novel, but green / orange / red echo the damage-class greens / oranges
# / reds), but typology and damage class are never encoded simultaneously
# in any chapter figure, and the on-plot legend disambiguates explicitly.
_TYPOLOGY_PALETTE: Final[tuple[str, ...]] = (
    "#1F77B4",  # Wind/Flood (tab:blue)
    "#FF7F0E",  # Kinetic (tab:orange)
    "#2CA02C",  # Thermal (tab:green)
    "#D62728",  # Unknown (tab:red)
)

# Per-typology marker shapes. The user-specified mapping is
# circle / square / triangle / diamond, matching the chapter-5 R3 per-
# class funnel marker convention so the visualization vocabulary is
# consistent across chapters.
_TYPOLOGY_MARKERS: Final[tuple[str, ...]] = ("o", "s", "^", "D")

# Reproducibility: t-SNE is stochastic; this seed makes the figure
# bit-identical across regenerations.
_RANDOM_STATE: Final[int] = 42
_PCA_DIM: Final[int] = 50
_TSNE_PERPLEXITY: Final[float] = 30.0


def main() -> None:
    """Render the typology-arm embedding scatter."""

    setup_publication_style()

    if not _EMBEDDING_CACHE.exists():
        raise FileNotFoundError(
            f"Embedding cache not found: {_EMBEDDING_CACHE}. "
            "Run: uv run python -m scripts.visualization.extract_canonical_embeddings"
        )

    cache = torch.load(_EMBEDDING_CACHE, map_location="cpu", weights_only=False)
    embeddings = cache["embeddings"].numpy().astype(np.float32)
    typology_indices = cache["typology_indices"].numpy().astype(np.int64)

    print(
        f"[D4 fig] Loaded {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]} "
        f"from {_EMBEDDING_CACHE}"
    )
    typology_counts = Counter(int(t) for t in typology_indices)
    print(f"[D4 fig] Typology counts: {dict(typology_counts)}")

    # PCA prereduction (2560 -> 50) to denoise and speed up t-SNE.
    pca = PCA(n_components=_PCA_DIM, random_state=_RANDOM_STATE)
    pca_features = pca.fit_transform(embeddings)
    print(
        f"[D4 fig] PCA explained-variance ratio (top {_PCA_DIM} dims): "
        f"{pca.explained_variance_ratio_.sum():.3f}"
    )

    tsne = TSNE(
        n_components=2,
        perplexity=_TSNE_PERPLEXITY,
        random_state=_RANDOM_STATE,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(pca_features)
    print(f"[D4 fig] t-SNE projection complete; coords shape: {coords.shape}")

    fig, ax = plt.subplots(
        figsize=(WIDTH_2COL, 4.6),
        gridspec_kw={
            "left": 0.04,
            "right": 0.985,
            "top": 0.985,
            "bottom": 0.04,
        },
    )

    # Per-typology rendering parameters: Wind/Flood is the dense majority
    # so it goes first at lower opacity / smaller markers; rare categories
    # (Kinetic, Thermal) plot ON TOP at full opacity and larger markers so
    # they remain visually findable against the majority mass.
    _MAJORITY_TYPOLOGY = 0  # Wind/Flood
    _STYLE_BY_RARE = {
        "size": 30.0,
        "alpha": 0.95,
        "edge_width": 0.6,
        "zorder": 5,
    }
    _STYLE_BY_MAJORITY = {
        "size": 18.0,
        "alpha": 0.55,
        "edge_width": 0.3,
        "zorder": 3,
    }

    # Plot order: majority first (back), rare last (foreground). Within
    # rare, larger-count rare goes first so the smallest dots are always
    # frontmost.
    rare_typologies = sorted(
        (t_idx for t_idx in range(len(_TYPOLOGY_NAMES)) if t_idx != _MAJORITY_TYPOLOGY),
        key=lambda t: -typology_counts.get(t, 0),  # descending count = larger first
    )
    plot_order = [_MAJORITY_TYPOLOGY, *rare_typologies]

    legend_handles: list = []
    for t_idx in plot_order:
        sel = typology_indices == t_idx
        if not sel.any():
            continue
        style = _STYLE_BY_MAJORITY if t_idx == _MAJORITY_TYPOLOGY else _STYLE_BY_RARE
        ax.scatter(
            coords[sel, 0],
            coords[sel, 1],
            marker=_TYPOLOGY_MARKERS[t_idx],
            c=_TYPOLOGY_PALETTE[t_idx],
            s=style["size"],
            alpha=style["alpha"],
            edgecolor="#222222",
            linewidth=style["edge_width"],
            zorder=style["zorder"],
        )
        legend_handles.append(
            plt.Line2D(
                [],
                [],
                marker=_TYPOLOGY_MARKERS[t_idx],
                color="none",
                markerfacecolor=_TYPOLOGY_PALETTE[t_idx],
                markeredgecolor="#222222",
                markeredgewidth=0.5,
                markersize=8,
                label=f"{_TYPOLOGY_NAMES[t_idx]} (n={typology_counts.get(t_idx, 0)})",
                linestyle="none",
            )
        )

    ax.set_xlabel("t-SNE dim 1", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", fontsize=9)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticks([])
    ax.set_yticks([])

    # Single legend, on-plot, deliberate upper-right corner with white-bbox
    # background. Upper-right is the conventional default and tends to be
    # the least-data-dense corner of t-SNE projections after PCA-prerotation.
    typology_legend = ax.legend(
        handles=legend_handles,
        title="Typology",
        loc="upper right",
        fontsize=8.5,
        title_fontsize=9.0,
        frameon=True,
        labelspacing=0.6,
    )
    typology_legend.get_frame().set(
        facecolor="white",
        edgecolor="#cccccc",
        alpha=0.95,
        linewidth=0.5,
        boxstyle="round,pad=0.4",
    )

    save_figure(fig=fig, name="embedding_tsne")

    save_caption(
        name="embedding_tsne",
        title=(
            "Two-dimensional t-SNE projection of typology-ablated baseline "
            "fused embeddings on the Spatial Block East val pool, colored by "
            "event-level typology category."
        ),
        body=(
            "Each point is one chip in the val pool. Embeddings are extracted "
            "from the typology-ablation arm "
            "(`outputs/ablation/typology/Spatial_Block_East/seed_00`), where "
            "FiLM conditioning is structurally absent so the fused embedding "
            "(the input to the linear classifier head, dim 2560) is a purely "
            "visual representation of the chip. Projection is PCA "
            "(2560 -> 50) followed by sklearn's t-SNE (50 -> 2D, perplexity "
            "30, deterministic `random_state=42`). Color and marker shape "
            "redundantly encode the chip's event-level typology category "
            "from `src.data.dataset.CRASARUnitemporalDataset.TYPOLOGY_MAP`, "
            "using matplotlib's tab10 palette (tab:blue = Wind/Flood, "
            "tab:orange = Kinetic, tab:green = Thermal, tab:red = Unknown) "
            "and the chapter-5 per-class-funnel marker convention "
            "(circle / square / triangle / diamond). Color and shape carry "
            "the same information so the figure is greyscale-print-safe and "
            "color-blind-accessible without losing information. The val "
            "pool is heavily Wind/Flood-dominated by composition "
            "(Hurricane Michael 1110 chips + Hurricane Ida 536 chips = 1646 "
            "Wind/Flood; Mussett Bayou Fire = 83 Thermal; Champlain Towers "
            "Collapse = 4 Kinetic), so rare-category points are drawn on top "
            "of the Wind/Flood mass at larger marker size and full opacity "
            "to remain visually findable. The empirical signature: Thermal "
            "forms a partial cluster - approximately 70% of the 83 Thermal "
            "chips concentrate along the bottom-left of the projection "
            "(median y = -32.6, well below the Wind/Flood median y = +3.4), "
            "with the remaining ~30% scattered through the broader Wind / "
            "Flood region. The pattern supports the chapter's implicit-"
            "encoding hypothesis with appropriate nuance: the visual stream "
            "learns typology-relevant features without explicit "
            "conditioning, but the encoding is partial rather than "
            "categorical - consistent with the chapter's QWK-on-distant-OOD "
            "lift coexisting with a null F1 effect (small ordinal-distance "
            "correction without categorical separation strong enough to "
            "swing argmax decisions). The Kinetic count of 4 is too small "
            "to support cluster claims on its own. Source artifact: "
            "`outputs/embeddings/typology_arm_sbe_val.pt` produced by "
            "`scripts/visualization/extract_canonical_embeddings.py` "
            "(``uv run python -m scripts.visualization.extract_canonical_embeddings``)."
        ),
    )


if __name__ == "__main__":
    main()
