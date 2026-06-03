"""Throwaway: render fig_mcdn_block and dump a PNG sidecar for visual review.

Re-runs the module's ``main`` after monkey-patching ``save_figure`` so the
SVG/PDF outputs are joined by a PNG preview. Always tracks the live module,
so it cannot drift out of sync with edits to ``fig_mcdn_block.main``.

Not registered in ``render_all`` and not consumed by the chapter or paper.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import scripts.visualization.fig_mcdn_block as M


def main() -> None:
    """Run the live MCDN-block render and additionally dump a PNG preview."""

    _orig_save = M.save_figure

    def _save_with_png(*, fig, name: str) -> None:
        _orig_save(fig=fig, name=name)
        out = Path("outputs/images") / f"_{name}_preview.png"
        fig.savefig(out, format="png", dpi=220,
                    bbox_inches="tight", pad_inches=0.05)
        print(f"Wrote {out}")

    M.save_figure = _save_with_png
    try:
        M.main()
    finally:
        M.save_figure = _orig_save


if __name__ == "__main__":
    main()
