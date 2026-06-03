"""Visualization scripts for the modeling chapter.

Each ``fig_<name>.py`` module is independently invokable via
``uv run python -m scripts.visualization.fig_<name>`` and renders a
publication-style SVG to ``outputs/images/<name>.svg`` (canonical) plus a
mirrored copy at ``doc/images/<name>.svg`` for Quarto chapter resolution.

Shared style defaults, constants, and the dual-write save helper live in
``_common.py``; see that module's docstring for the publication-style contract.
"""
