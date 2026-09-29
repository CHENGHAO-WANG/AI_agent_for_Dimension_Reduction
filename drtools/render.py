"""Markdown to PDF, and the two installs that have to be there.

pandoc and a PDF engine are separate programs with separate fixes, and pandoc's own
error for the second is about a missing engine rather than a missing program. Naming
them separately is the difference between an agent installing the right thing and
installing pandoc a second time.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from drtools.contract import ContractError

PDF_ENGINES = ("xelatex", "lualatex", "pdflatex", "tectonic", "typst", "weasyprint")
"""Engines pandoc can drive, in the order this toolbox prefers them.

The two that read Unicode come first. Found on day 21: pdflatex refused a minus sign,
U+2212, in the agent's prose, which xelatex renders without a warning.
"""


def render_pdf(source: Path, destination: Path) -> dict[str, Any]:
    """Render one Markdown file, refusing rather than raising when a program is absent."""
    if shutil.which("pandoc") is None:
        raise ContractError(
            "pandoc is not on PATH, and it is what turns the report into a PDF. "
            "Install it from https://pandoc.org/installing.html. Nothing is lost "
            f"meanwhile: the Markdown at {source} is the source of truth and is "
            "already complete; the PDF is a rendering of it."
        )

    engine = next((name for name in PDF_ENGINES if shutil.which(name)), None)
    if engine is None:
        raise ContractError(
            "pandoc is installed but no PDF engine is, and they are two different "
            "programs: pandoc reads the Markdown, a separate engine makes the PDF. "
            f"Install one of {', '.join(PDF_ENGINES)} — a TeX distribution such as "
            "MiKTeX or TeX Live provides pdflatex."
        )

    source, destination = source.resolve(), destination.resolve()
    completed = subprocess.run(
        ["pandoc", str(source), "-o", str(destination), f"--pdf-engine={engine}"],
        capture_output=True,
        # pandoc writes UTF-8 whatever the console's code page. Read as GBK on day 21,
        # its error message failed to decode and the refusal lost its reason.
        encoding="utf-8",
        errors="replace",
        # Images are referenced relative to the results folder, which is where the
        # Markdown lives, so that is where pandoc has to resolve them from.
        cwd=str(source.parent),
    )
    if completed.returncode != 0:
        raise ContractError(
            f"pandoc could not render {source.name} with {engine}: "
            f"{completed.stderr.strip()[:500]}"
        )
    return {"path": str(destination), "engine": engine}
