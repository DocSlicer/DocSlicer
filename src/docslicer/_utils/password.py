"""Password candidate generation and document decryption helpers."""

from __future__ import annotations

import io
import re
from datetime import date
from pathlib import Path

_CURRENT_YEAR = date.today().year
_YEAR_RANGE = [str(y) for y in range(_CURRENT_YEAR - 2, _CURRENT_YEAR + 2)]

_DOC_PASSWORDS: list[str] = [
    "",
    " ",
    # Excel default sheet-protection password
    "VelvetSweatshop",
    # generic
    "password", "Password", "PASSWORD", "pass", "Pass@123", "admin123", 
    "qwerty", "Qwerty",
    "admin", "Admin", "administrator", "Administrator",
    "demo", "Demo",
    "document", "Document",
    "open", "Open",
    "compensation", "Compensation", "comp", "Comp",
    "model", "Model",
    "presentation", "Presentation",
    "report", "Report",
    "1234", "12345", "123456", "0000", "1111", "123456789",
    "welcome",
    # bare years
    *_YEAR_RANGE,
    # corporate root + year combinations
    *(
        f"{root}{year}"
        for root in (
            "Confidential", "confidential",
            "Private", "private",
            "Internal", "internal",
            "Archive", "archive",
            "Financial", "financial",
            "Report", "report",
        )
        for year in _YEAR_RANGE
    ),
]


def _password_candidates(
    password: str | None,
    source_filename: str | None = None,
) -> list[str]:
    """Return an ordered list of passwords to try, starting with the user-supplied one."""
    seen: set[str] = set()
    candidates: list[str] = []

    def _add(pw: str) -> None:
        if pw not in seen:
            seen.add(pw)
            candidates.append(pw)

    if password is not None:
        _add(password)

    if source_filename:
        stem = Path(source_filename).stem
        _add(stem)
        _add(stem.lower())
        _add(stem.upper())
        for part in stem.replace("-", " ").replace("_", " ").split():
            _add(part)
            _add(part.lower())
            _add(part.upper())
        for token in re.findall(r"\d{4,8}", stem):
            _add(token)

    for pw in _DOC_PASSWORDS:
        _add(pw)

    return candidates


def decrypt_pdf(
    pdf_bytes: bytes,
    password: str | None = None,
    source_filename: str | None = None,
) -> bytes:
    """Return decrypted PDF bytes, trying password candidates until one works.

    Raises ValueError if no candidate succeeds.
    """
    import pikepdf

    candidates = _password_candidates(password, source_filename)
    buf = io.BytesIO(pdf_bytes)
    for pw in candidates:
        buf.seek(0)
        try:
            with pikepdf.open(buf, password=pw) as pdf:
                out = io.BytesIO()
                pdf.save(out)
                return out.getvalue()
        except pikepdf.PasswordError:
            continue
    raise ValueError(
        "Could not open password-protected PDF: all candidate passwords failed. "
        "Pass the correct password via password=."
    )


def decrypt_office(
    data: bytes,
    password: str | None = None,
    source_filename: str | None = None,
) -> bytes:
    """Return decrypted Office (DOCX/XLSX/PPTX) bytes, trying password candidates.

    Requires msoffcrypto-tool (pip install 'docslicer[crypto]').
    Raises ValueError if no candidate succeeds.
    """
    try:
        import msoffcrypto
    except ImportError:
        raise ImportError(
            "Decrypting password-protected Office files requires msoffcrypto-tool. "
            "Install it with: pip install 'docslicer[crypto]'"
        )

    candidates = _password_candidates(password, source_filename)
    for pw in candidates:
        try:
            f = io.BytesIO(data)
            office = msoffcrypto.OfficeFile(f)
            if not office.is_encrypted():
                return data
            out = io.BytesIO()
            office.load_key(password=pw)
            office.decrypt(out)
            return out.getvalue()
        except Exception:
            continue
    raise ValueError(
        "Could not decrypt password-protected Office file: all candidate passwords failed. "
        "Pass the correct password via password=."
    )


_OLE_MAGIC = b"\xD0\xCF\x11\xE0"


def is_encrypted_office(data: bytes) -> bool:
    """True when the bytes look like an OLE container (encrypted Office file)."""
    return data[:4] == _OLE_MAGIC
