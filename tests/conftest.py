import pytest
from pathlib import Path

import docslicer

SAMPLES = Path(__file__).parent.parent / "examples" / "sample_docs"


@pytest.fixture(scope="session")
def pdf_result():
    return docslicer.parse_document(SAMPLES / "financial_report.pdf")


@pytest.fixture(scope="session")
def scanned_result():
    try:
        return docslicer.parse_document(SAMPLES / "letter_scanned.pdf")
    except Exception as e:
        pytest.skip(f"OCR not available or failed: {e}")


@pytest.fixture(scope="session")
def academic_result():
    return docslicer.parse_document(SAMPLES / "academic_paper.pdf")


@pytest.fixture(scope="session")
def html_result():
    return docslicer.parse_document(SAMPLES / "sec_10q.html")


@pytest.fixture(scope="session")
def docx_result():
    return docslicer.parse_document(SAMPLES / "infosec_policy.docx")


@pytest.fixture(scope="session")
def pptx_result():
    return docslicer.parse_document(SAMPLES / "financial_review.pptx")
