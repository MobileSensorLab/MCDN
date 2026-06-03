"""Generate the project coverage badge as a self-contained shields-style SVG.

Parses a coverage.py Cobertura XML report, computes the combined line+branch
coverage (matching the genbadge total formula), embeds the pytest logo, and renders
a ``[ (pytest) PyTest | <pct>% ]`` badge via pybadges. Runs identically on a developer
machine and in the GitLab CI runner, and needs no network access (the logo is inlined
as a base64 data URI rather than fetched).
"""

import argparse
import base64
import importlib.util
import sys
import types
import warnings
from pathlib import Path
from xml.etree import ElementTree

# pybadges imports pkg_resources, which emits a deprecation warning; keep CI logs clean.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

# pybadges 3.0.1 imports the stdlib `imghdr` module, which was removed in Python 3.13. The
# pytest logo is passed as an inline SVG data URI (never a raster image), so pybadges never
# calls imghdr; a no-op shim satisfies the import without adding a backport dependency. Only
# shim when imghdr is genuinely absent (3.13+), leaving the real module intact on 3.12.
# Remove this once pybadges ships a 3.13-compatible release.
if importlib.util.find_spec("imghdr") is None:
    _imghdr_shim = types.ModuleType("imghdr")
    _imghdr_shim.what = lambda *args, **kwargs: None  # noqa: ARG005
    sys.modules["imghdr"] = _imghdr_shim

from pybadges import badge  # noqa: E402  (imported after the imghdr shim + warning filter)

# genbadge coverage bands and hexes, preserved for visual continuity with prior badges.
COLOR_HEX = {"brightgreen": "#4c1", "green": "#97ca00", "orange": "#fe7d37", "red": "#e05d44"}

DEFAULT_INPUT = Path(".pytest_cache/coverage.xml")
DEFAULT_OUTPUT = Path("data/readme/coverage.svg")
DEFAULT_LOGO = Path("data/readme/pytest.svg")


def total_coverage(xml_path: Path) -> float:
    """Return combined line+branch coverage percent from a Cobertura XML report.

    Mirrors genbadge's total: ``(lines_covered + branches_covered) /
    (lines_valid + branches_valid) * 100``.

    Args:
        xml_path: Path to the coverage.py XML report.
    """

    root = ElementTree.parse(xml_path).getroot()
    lines_covered = int(root.attrib["lines-covered"])
    lines_valid = int(root.attrib["lines-valid"])
    branches_covered = int(root.attrib["branches-covered"])
    branches_valid = int(root.attrib["branches-valid"])

    denominator = lines_valid + branches_valid
    return (lines_covered + branches_covered) / denominator * 100 if denominator else 0.0


def coverage_color(pct: float) -> str:
    """Return the badge hex color for a coverage percentage using genbadge's bands."""

    if pct < 50:
        return COLOR_HEX["red"]
    if pct < 75:
        return COLOR_HEX["orange"]
    if pct < 90:
        return COLOR_HEX["green"]
    return COLOR_HEX["brightgreen"]


# The pytest artwork occupies only the central region of its 0 0 128 128 canvas, so at the
# badge's fixed 14px logo box it looks underweight beside the python/PyTorch logos (which fill
# their canvases). Cropping the embedded copy to the artwork's bounding box (a centered square
# with a few units of breathing room) makes the logo fill the box and match the other badges'
# visual weight, without touching the badge geometry/spacing or the pytest.svg asset itself.
PYTEST_SOURCE_VIEWBOX = 'viewBox="0 0 128 128"'
PYTEST_TIGHT_VIEWBOX = 'viewBox="20.66 21.32 83.01 83.01"'


def logo_data_uri(logo_path: Path) -> str:
    """Return a base64 ``data:image/svg+xml`` URI for an SVG logo file.

    For the pytest logo the canvas is tightened to the artwork bounds so the mark renders at
    a weight consistent with the other badges; any other logo is embedded unchanged.
    """

    svg_text = logo_path.read_text(encoding="utf-8")
    svg_text = svg_text.replace(PYTEST_SOURCE_VIEWBOX, PYTEST_TIGHT_VIEWBOX)
    encoded = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def build_badge(input_file: Path, output_file: Path, logo_file: Path) -> None:
    """Render the PyTest coverage badge SVG and write it to ``output_file``."""

    pct = total_coverage(input_file)
    svg = badge(
        left_text="PyTest",
        right_text=f"{round(pct)}%",
        right_color=coverage_color(pct),
        logo=logo_data_uri(logo_file),
        embed_logo=False
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(svg, encoding="utf-8")
    print(f"Wrote {output_file}  ->  PyTest | {round(pct)}%  ({pct:.2f}% raw)")


def main() -> None:
    """Parse CLI arguments and generate the coverage badge."""

    parser = argparse.ArgumentParser(description="Generate the PyTest coverage badge SVG.")
    parser.add_argument("-i", "--input-file", type=Path, default=DEFAULT_INPUT, help="Coverage XML report path.")
    parser.add_argument("-o", "--output-file", type=Path, default=DEFAULT_OUTPUT, help="Output SVG badge path.")
    parser.add_argument("--logo", type=Path, default=DEFAULT_LOGO, help="SVG logo file to embed.")
    args = parser.parse_args()

    build_badge(args.input_file, args.output_file, args.logo)


if __name__ == "__main__":
    main()
