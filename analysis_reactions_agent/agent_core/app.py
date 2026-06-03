from __future__ import annotations

import argparse
import sys
import time
import urllib.error
from pathlib import Path
from typing import Iterable

from local_reaction_evidence import build_local_evidence

from .chemistry import rdkit_is_available
from .config import PROVIDER_DEFAULTS, load_runtime_config, resolve_runtime_options
from .llm import LLMClient
from .logging_utils import log_progress, write_debug_snapshot
from .pipeline import analyze_batches
from .records import build_reaction_record, iter_rows, load_existing_results, now_iso, write_outputs


DEFAULT_INPUT_CSV = Path("data/demo_reactions.csv")
DEFAULT_OUTPUT_CSV = Path("data/demo_reactions_ranked.csv")
DEFAULT_OUTPUT_JSONL = Path("data/demo_reactions_raw.jsonl")
DEFAULT_PROGRESS_LOG = Path("data/demo_reactions_feasibility_progress.txt")
DEFAULT_CONFIG_PATH = Path("llm_provider_config.json")
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_WORKERS = 1
DEFAULT_KEEP_THRESHOLD = 60.0
DEFAULT_HEARTBEAT_SECONDS = 30
DEFAULT_CSV_SNAPSHOT_SECONDS = 60
DEFAULT_PROVIDER = "gemini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based chemical reaction feasibility agent")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--progress-log", type=Path, default=DEFAULT_PROGRESS_LOG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--provider", default="", choices=sorted(PROVIDER_DEFAULTS.keys()))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--keep-threshold", type=float, default=DEFAULT_KEEP_THRESHOLD)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--heartbeat-seconds", type=int, default=DEFAULT_HEARTBEAT_SECONDS)
    parser.add_argument("--csv-snapshot-seconds", type=int, default=DEFAULT_CSV_SNAPSHOT_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--debug-notes", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not rdkit_is_available():
        raise RuntimeError("RDKit is required. Run this agent in the 'yieldnet-reaction-agent' environment.")
    if not args.input.exists():
        raise FileNotFoundError("Input CSV does not exist: {0}".format(args.input))
    if args.batch_size <= 0 or args.max_workers <= 0 or args.heartbeat_seconds <= 0:
        raise ValueError("Invalid positive integer runtime option")
    if args.csv_snapshot_seconds < 0:
        raise ValueError("--csv-snapshot-seconds must be >= 0")
    if not args.model:
        raise ValueError("No model resolved for provider '{0}'".format(args.provider))
    if not args.api_url:
        raise ValueError("No API URL resolved for provider '{0}'".format(args.provider))
    if not args.dry_run and not args.api_key:
        raise ValueError("Set {0}, create {1}, or pass --api-key".format(args.api_key_env_var, args.api_key_file))


def is_temporary_path(path: Path) -> bool:
    return path.name.startswith("_tmp_")


def cleanup_temporary_outputs(paths: Iterable[Path]) -> None:
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not is_temporary_path(path):
            continue
        seen.add(resolved)
        if path.exists():
            path.unlink()


def main(script_dir: Path) -> int:
    args = parse_args()
    runtime_config = load_runtime_config(script_dir, args.config)
    args = resolve_runtime_options(args, runtime_config, script_dir, DEFAULT_PROVIDER)
    validate_args(args)

    rows = list(iter_rows(args.input, offset=args.offset, limit=args.limit))
    rows_by_id = {index + args.offset: dict(row) for index, row in enumerate(rows)}
    total_target = len(rows_by_id)
    all_records = [build_reaction_record(row_id, row) for row_id, row in rows_by_id.items()]
    local_evidence_map = build_local_evidence(all_records)
    existing_results = {} if args.no_resume else load_existing_results(args.output_jsonl)
    existing_results = {
        row_id: row
        for row_id, row in existing_results.items()
        if "predicted_condition_summary" not in row and "predicted_condition_steps" not in row
    }
    existing_row_ids = set(row_id for row_id in existing_results if row_id in rows_by_id)
    records = []
    for record in all_records:
        row_id = int(record["row_id"])
        record["structural_evidence_summary"] = local_evidence_map.get(row_id, {}).get("summary", "")
        record["inference_rules"] = [
            "Only the major product may be written.",
            "Missing side products are allowed.",
            "Missing auxiliary reagents/conditions must not be treated as impossibility.",
        ]
        if row_id not in existing_row_ids:
            records.append(record)

    if args.debug_notes:
        write_debug_snapshot(args.debug_notes, total_target, len(records), local_evidence_map)
    log_progress(
        args.progress_log,
        "[{timestamp}] start total={total} already_completed={done} pending={pending} batch_size={batch_size} max_workers={workers} model={model}".format(
            timestamp=now_iso(),
            total=total_target,
            done=len(existing_row_ids),
            pending=len(records),
            batch_size=args.batch_size,
            workers=args.max_workers,
            model=args.model,
        ),
    )

    client = None
    if not args.dry_run:
        client = LLMClient(
            provider=args.provider,
            api_key=args.api_key,
            api_url=args.api_url,
            model=args.model,
            timeout=args.timeout,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    started_at = time.time()
    temp_outputs = [
        args.output_csv,
        args.output_jsonl,
        args.progress_log,
    ]
    if args.debug_notes:
        temp_outputs.append(args.debug_notes)
    try:
        results = []
        if records:
            results = analyze_batches(
                records,
                rows_by_id,
                local_evidence_map,
                client,
                args.batch_size,
                args.max_workers,
                args.keep_threshold,
                args.max_retries,
                args.dry_run,
                args.output_jsonl,
                args.output_csv,
                args.progress_log,
                len(existing_row_ids),
                total_target,
                args.heartbeat_seconds,
                args.csv_snapshot_seconds,
                existing_results,
            )

        merged_results = dict(existing_results)
        for row in results:
            merged_results[int(row["row_id"])] = row
        final_rows = [merged_results[row_id] for row_id in sorted(merged_results) if row_id in rows_by_id]
        write_outputs(args.output_csv, args.output_jsonl, final_rows)
        duration = time.time() - started_at
        log_progress(
            args.progress_log,
            "[{timestamp}] finished total={total} newly_processed={new} duration_s={duration:.2f}".format(
                timestamp=now_iso(),
                total=len(final_rows),
                new=len(results),
                duration=duration,
            ),
        )
        print("Processed {0} reactions, wrote {1} and {2} in {3:.1f}s".format(len(final_rows), args.output_csv, args.output_jsonl, duration))
        return 0
    except urllib.error.HTTPError as exc:
        sys.stderr.write("HTTP error from API: {0}\n".format(exc.read().decode("utf-8", errors="replace")))
        return 1
    except Exception as exc:
        sys.stderr.write("Analysis failed: {0}\n".format(exc))
        return 1
    finally:
        cleanup_temporary_outputs(temp_outputs)
