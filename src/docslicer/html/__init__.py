# Lazy imports to avoid loading orchestrator when importing individual steps
# Use: from docslicer.html import run_pipeline

def __getattr__(name):
    if name == "run_pipeline":
        from .html_orchestrator import run_pipeline
        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "run_pipeline",
]

