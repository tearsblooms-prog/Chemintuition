from __future__ import annotations

import http.client
import socket
import urllib.error
from urllib.parse import urlparse
from pathlib import Path
from typing import Dict, Sequence

from .records import clean_text, now_iso


def append_progress_log(progress_log: Path, message: str) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    with progress_log.open("a", encoding="utf-8") as outfile:
        outfile.write(message.rstrip() + "\n")


def log_progress(progress_log: Path, message: str) -> None:
    print(message, flush=True)
    append_progress_log(progress_log, message)


def summarize_exception(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = ""
        if detail:
            return "{0}: {1}".format(exc, clean_text(detail, 400))
    return str(exc)


def classify_exception(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return "http_error"
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, socket.gaierror):
            return "dns_error"
        if isinstance(reason, TimeoutError):
            return "timeout_error"
        if isinstance(reason, OSError):
            return "network_os_error"
        return "url_error"
    if isinstance(exc, socket.gaierror):
        return "dns_error"
    if isinstance(exc, TimeoutError):
        return "timeout_error"
    if isinstance(exc, http.client.IncompleteRead):
        return "incomplete_read"
    if isinstance(exc, OSError):
        return "network_os_error"
    return "application_error"


def is_retryable_exception(exc: Exception) -> bool:
    return classify_exception(exc) in {
        "dns_error",
        "timeout_error",
        "network_os_error",
        "url_error",
        "incomplete_read",
        "http_error",
    }


def retry_delay_seconds(exc: Exception, attempt: int) -> int:
    error_type = classify_exception(exc)
    if error_type == "dns_error":
        return min(15 * attempt, 60)
    if error_type in {"timeout_error", "network_os_error", "url_error", "incomplete_read"}:
        return min(5 * attempt, 30)
    return min(2 ** attempt, 8)


def diagnose_endpoint(api_url: str, model: str = "") -> str:
    try:
        formatted_url = api_url.format(model=model) if model else api_url
        parsed = urlparse(formatted_url)
        host = parsed.hostname
        if not host:
            return "endpoint=unparseable"
        addresses = sorted(set(item[4][0] for item in socket.getaddrinfo(host, parsed.port or 443)))
        return "endpoint_host={0} resolved={1}".format(host, ",".join(addresses[:4]))
    except Exception as exc:
        return "endpoint_check_failed={0}".format(clean_text(str(exc), 200))


def append_debug_notes(debug_notes: Path, message: str) -> None:
    debug_notes.parent.mkdir(parents=True, exist_ok=True)
    with debug_notes.open("a", encoding="utf-8") as outfile:
        outfile.write(message.rstrip() + "\n")


def write_debug_snapshot(
    debug_notes: Path,
    total_rows: int,
    pending_rows: int,
    local_evidence_map: Dict[int, Dict[str, object]],
) -> None:
    lines = [
        "[{timestamp}] task snapshot".format(timestamp=now_iso()),
        "Goal: judge main-product feasibility even when only the major product is written and reaction conditions are incomplete or missing.",
        "Constraints: do not reject only because side products are omitted; evidence must come from the single reaction itself rather than from matching answers in the corpus.",
        "Input rows: {0}".format(total_rows),
        "Pending rows this run: {0}".format(pending_rows),
        "Evidence highlights:",
    ]
    for row_id in sorted(local_evidence_map):
        evidence = local_evidence_map[row_id]
        lines.append(
            "row_id={0}; family={1}; score_floor={2}; keep_floor={3}; summary={4}".format(
                row_id,
                evidence.get("family", ""),
                evidence.get("score_floor", 0.0),
                evidence.get("keep_floor", False),
                evidence.get("summary", ""),
            )
        )
    lines.append("Next: run the agent, inspect low-score outliers, and iterate on generic structure rules or real cheminformatics tools.")
    append_debug_notes(debug_notes, "\n".join(lines) + "\n")
