from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .llm import mock_analyze_batch, normalize_analysis
from .logging_utils import (
    classify_exception,
    diagnose_endpoint,
    is_retryable_exception,
    log_progress,
    retry_delay_seconds,
    summarize_exception,
)
from .records import append_jsonl_rows, now_iso, write_ranked_csv


def batched(items: Sequence[Dict[str, object]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def analyze_batches(
    records: Sequence[Dict[str, object]],
    rows_by_id: Dict[int, Dict[str, str]],
    local_evidence_map: Dict[int, Dict[str, object]],
    client,
    batch_size: int,
    max_workers: int,
    keep_threshold: float,
    max_retries: int,
    dry_run: bool,
    output_jsonl: Path,
    output_csv: Path,
    progress_log: Path,
    already_completed: int,
    total_target: int,
    heartbeat_seconds: int,
    csv_snapshot_seconds: int,
    existing_results: Dict[int, Dict[str, object]],
) -> List[Dict[str, object]]:
    normalized_results = []
    batches = list(batched(records, batch_size))
    write_lock = threading.Lock()
    completed_count = already_completed
    active_batches: Dict[int, Dict[str, object]] = {}
    last_completed_at = time.time()
    last_csv_snapshot_at = time.time()

    def run_single_batch(batch_index: int, batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        attempts = 0
        batch_started = time.time()
        batch_row_ids = [int(item["row_id"]) for item in batch]
        while True:
            attempts += 1
            with write_lock:
                active_batches[batch_index] = {
                    "row_ids": batch_row_ids,
                    "attempt": attempts,
                    "attempt_started_at": time.time(),
                }
                log_progress(
                    progress_log,
                    "[{timestamp}] batch={batch}/{batch_total} status=started attempt={attempt}/{max_retries} row_ids={start}-{end} size={size}".format(
                        timestamp=now_iso(),
                        batch=batch_index,
                        batch_total=len(batches),
                        attempt=attempts,
                        max_retries=max_retries,
                        start=min(batch_row_ids),
                        end=max(batch_row_ids),
                        size=len(batch_row_ids),
                    ),
                )
            try:
                raw_reactions = mock_analyze_batch(batch, keep_threshold) if dry_run else client.analyze_batch(batch)
                raw_by_id = {int(item["row_id"]): item for item in raw_reactions if "row_id" in item}
                normalized = []
                for source in batch:
                    row_id = int(source["row_id"])
                    if row_id not in raw_by_id:
                        raise ValueError("Missing row_id {0} in model response".format(row_id))
                    normalized.append(
                        normalize_analysis(raw_by_id[row_id], rows_by_id[row_id], keep_threshold, local_evidence_map.get(row_id))
                    )
                return {
                    "batch_index": batch_index,
                    "row_ids": batch_row_ids,
                    "attempts": attempts,
                    "started_at": batch_started,
                    "ended_at": time.time(),
                    "rows": normalized,
                }
            except Exception as exc:
                error_kind = classify_exception(exc)
                error_summary = summarize_exception(exc)
                if error_kind in {"dns_error", "timeout_error", "network_os_error", "url_error", "incomplete_read"}:
                    error_summary = "{0} [{1}]".format(
                        error_summary,
                        diagnose_endpoint(getattr(client, "api_url", ""), getattr(client, "model", "")),
                    )
                if attempts >= max_retries or not is_retryable_exception(exc):
                    with write_lock:
                        active_batches.pop(batch_index, None)
                        log_progress(
                            progress_log,
                            "[{timestamp}] batch={batch}/{batch_total} status=failed attempt={attempt}/{max_retries} row_ids={start}-{end} error_kind={error_kind} error={error}".format(
                                timestamp=now_iso(),
                                batch=batch_index,
                                batch_total=len(batches),
                                attempt=attempts,
                                max_retries=max_retries,
                                start=min(batch_row_ids),
                                end=max(batch_row_ids),
                                error_kind=error_kind,
                                error=error_summary,
                            ),
                        )
                    raise RuntimeError("Batch failed after {0} attempts: {1}".format(attempts, exc))
                delay_seconds = retry_delay_seconds(exc, attempts)
                with write_lock:
                    log_progress(
                        progress_log,
                        "[{timestamp}] batch={batch}/{batch_total} status=retrying attempt={attempt}/{max_retries} row_ids={start}-{end} error_kind={error_kind} sleep_s={delay} error={error}".format(
                            timestamp=now_iso(),
                            batch=batch_index,
                            batch_total=len(batches),
                            attempt=attempts,
                            max_retries=max_retries,
                            start=min(batch_row_ids),
                            end=max(batch_row_ids),
                            error_kind=error_kind,
                            delay=delay_seconds,
                            error=error_summary,
                        ),
                    )
                # Reset the global opener after transport errors so the next attempt starts fresh.
                if isinstance(exc, urllib.error.URLError):
                    urllib.request.install_opener(urllib.request.build_opener())
                time.sleep(delay_seconds)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_single_batch, batch_index, batch): batch_index for batch_index, batch in enumerate(batches, 1)}
        pending_futures = set(future_map)
        while pending_futures:
            done_futures, pending_futures = wait(pending_futures, timeout=max(1, heartbeat_seconds), return_when=FIRST_COMPLETED)
            if not done_futures:
                with write_lock:
                    active_descriptions = []
                    for batch_index in sorted(active_batches):
                        active = active_batches[batch_index]
                        active_descriptions.append(
                            "batch={0} row_ids={1}-{2} attempt={3} active_s={4:.1f}".format(
                                batch_index,
                                min(active["row_ids"]),
                                max(active["row_ids"]),
                                active["attempt"],
                                time.time() - float(active["attempt_started_at"]),
                            )
                        )
                    if not active_descriptions:
                        active_descriptions.append("no_active_batches_visible")
                    log_progress(
                        progress_log,
                        "[{timestamp}] heartbeat completed={completed}/{total} pending_batches={pending} idle_s={idle:.1f} active={active}".format(
                            timestamp=now_iso(),
                            completed=completed_count,
                            total=total_target,
                            pending=len(pending_futures),
                            idle=time.time() - last_completed_at,
                            active="; ".join(active_descriptions),
                        ),
                    )
                continue
            for future in done_futures:
                batch_index = future_map[future]
                with write_lock:
                    active_batches.pop(batch_index, None)
                batch_result = future.result()
                batch_rows = batch_result["rows"]
                with write_lock:
                    append_jsonl_rows(output_jsonl, batch_rows)
                    normalized_results.extend(batch_rows)
                    completed_count += len(batch_rows)
                    last_completed_at = time.time()
                    if csv_snapshot_seconds > 0 and time.time() - last_csv_snapshot_at >= csv_snapshot_seconds:
                        snapshot_rows = dict(existing_results)
                        for row in normalized_results:
                            snapshot_rows[int(row["row_id"])] = row
                        write_ranked_csv(output_csv, list(snapshot_rows.values()))
                        last_csv_snapshot_at = time.time()
                        log_progress(progress_log, "[{timestamp}] csv_snapshot rows={rows} path={path}".format(timestamp=now_iso(), rows=len(snapshot_rows), path=output_csv))
                    log_progress(
                        progress_log,
                        "[{timestamp}] batch={batch}/{batch_total} status=completed row_ids={start}-{end} size={size} attempts={attempts} duration_s={duration:.2f} completed={completed}/{total}".format(
                            timestamp=now_iso(),
                            batch=batch_result["batch_index"],
                            batch_total=len(batches),
                            start=min(batch_result["row_ids"]),
                            end=max(batch_result["row_ids"]),
                            size=len(batch_rows),
                            attempts=batch_result["attempts"],
                            duration=batch_result["ended_at"] - batch_result["started_at"],
                            completed=completed_count,
                            total=total_target,
                        ),
                    )
    return normalized_results
