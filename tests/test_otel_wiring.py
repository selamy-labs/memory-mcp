"""OpenTelemetry instrumentation wiring for memory-mcp (#743).

memory-mcp is instrumented via OpenTelemetry zero-code auto-instrumentation
(`opentelemetry-instrument memory-mcp`), kept vendor-neutral via OTEL_* env.
These tests lock the wiring: the `otel` extra declares the distro + an OTLP
exporter, and the README documents the wrapped start command plus the
stdio-safety contract (OTLP only — a console/stdout exporter would corrupt the
JSON-RPC stdio stream).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _otel_extra_block() -> str:
    start = PYPROJECT.index("otel = [")
    end = PYPROJECT.index("]", start)
    return PYPROJECT[start:end]


def test_otel_extra_declares_distro_and_otlp_exporter() -> None:
    block = _otel_extra_block()
    assert "opentelemetry-distro" in block  # provides `opentelemetry-instrument`
    assert "opentelemetry-exporter-otlp" in block


def test_readme_documents_opentelemetry_instrument_startup() -> None:
    assert "opentelemetry-instrument memory-mcp" in README
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in README


def test_readme_warns_against_stdout_exporter_for_stdio_safety() -> None:
    lowered = README.lower()
    assert "stdio safety" in lowered
    assert "never" in lowered and "console" in lowered
    assert "otlp only" in lowered
