"""Shared chart styling for ezpz plots.

Applies the ``ambivalent`` matplotlib stylesheet and registers Iosevka
(downloaded to ``~/.local/share/fonts/Iosevka/``) so every chart uses
the same monospace face.

Import side-effect: calling ``apply_style()`` is idempotent. The
caller imports this module before creating any figures.

Reproduce font setup once per machine (login node, with proxy):
    URL=https://github.com/be5invis/Iosevka/releases/download/v34.6.1/PkgTTC-Iosevka-34.6.1.zip
    mkdir -p ~/.local/share/fonts/Iosevka
    curl -L -o /tmp/iosevka.zip --proxy http://proxy.alcf.anl.gov:3128 $URL
    unzip -q -o /tmp/iosevka.zip -d ~/.local/share/fonts/Iosevka/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt

import ambivalent

_IOSEVKA_DIR = Path.home() / ".local/share/fonts/Iosevka"

_applied = False


def apply_style(font_family: str | list[str] = "Iosevka") -> None:
    """Apply ambivalent style + Iosevka font (idempotent)."""
    global _applied
    if _applied:
        return
    if _IOSEVKA_DIR.is_dir():
        for f in _IOSEVKA_DIR.iterdir():
            if f.suffix.lower() in (".ttf", ".ttc", ".otf"):
                _fm.fontManager.addfont(str(f))
    plt.style.use(ambivalent.STYLES["ambivalent"])
    if isinstance(font_family, str):
        font_family = [font_family, "DejaVu Sans Mono", "monospace"]
    plt.rcParams["font.family"] = font_family
    _applied = True
