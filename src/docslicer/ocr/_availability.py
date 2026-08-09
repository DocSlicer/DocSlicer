# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Cheap probe for whether the OCR pipeline can actually run.

OCR needs two things that arrive by different routes: the `ocr` extra supplies
the Python packages, while the Tesseract engine and its traineddata come from a
system install pip cannot provide. Missing either one surfaces deep in the
pipeline as something the caller cannot act on — `ImportError: No module named
'cv2'` from a step module, or the tessdata `RuntimeError` from
step_01_word_extractor. Probing both up front turns those into one message that
names the missing piece.
"""

from __future__ import annotations

import os
import shutil
from importlib.util import find_spec


class OcrUnavailableError(RuntimeError):
    """A document needs OCR to be read, but this install cannot run OCR.

    Distinct from an OCR failure: nothing was attempted. The document is
    parseable, just not here — so the message carries what is missing and how
    to install it, and callers handling many documents can report this one and
    continue rather than treating it as a broken parse.
    """


# Import name -> the distribution `docslicer[ocr]` installs it from.
_MODULES = {"cv2": "opencv-python", "tesserocr": "tesserocr", "PIL": "Pillow"}


def ocr_unavailable_reason() -> str | None:
    """Return why OCR cannot run here, as a sentence fragment, or None if it can.

    Deliberately never imports the OCR modules: `find_spec` answers the
    question, and importing cv2 and tesserocr is expensive enough that a parse
    which will not use them should not pay for it.
    """
    missing = [dist for mod, dist in _MODULES.items() if find_spec(mod) is None]
    if missing:
        return (
            f"the OCR extra is not installed (missing {', '.join(missing)}). "
            "Install it with: pip install 'docslicer[ocr]'."
        )

    # Mirrors _resolve_tessdata_path in step_01_word_extractor: the prebuilt
    # tesserocr wheel bundles a libtesseract that does not know where
    # traineddata lives, so we need either an explicit path or a system
    # `tesseract` binary to ask. Same condition, checked before the import.
    if not (os.environ.get("TESSDATA_PREFIX") or shutil.which("tesseract")):
        return (
            "the Tesseract engine is not installed, so no traineddata can be "
            "located. Install it (brew install tesseract, or apt install "
            "tesseract-ocr), or set TESSDATA_PREFIX to the directory holding "
            "eng.traineddata."
        )

    return None
