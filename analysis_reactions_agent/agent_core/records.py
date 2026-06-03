from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def parse_float(value: Optional[str], default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def now_iso() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def iter_rows(input_csv: Path, offset: int = 0, limit: int = 0) -> Iterable[Dict[str, str]]:
    with input_csv.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for index, row in enumerate(reader):
            if index < offset:
                continue
            if limit and index >= offset + limit:
                break
            yield row


def build_reaction_record(row_id: int, row: Dict[str, str]) -> Dict[str, object]:
    reactant1 = (row.get("reactant1_smiles") or "").strip()
    reactant2 = (row.get("reactant2_smiles") or "").strip()
    product = (row.get("product_smiles") or "").strip()
    predicted_yield = parse_float(row.get("predicted_yield"), default=0.0)
    reaction_smiles = ">".join([".".join([s for s in [reactant1, reactant2] if s]), "", product])
    return {
        "row_id": row_id,
        "reaction_smiles": reaction_smiles,
        "reactant1_smiles": reactant1,
        "reactant2_smiles": reactant2,
        "product_smiles": product,
        "predicted_yield": predicted_yield,
        "catalyst": (row.get("catalyst") or "").strip(),
        "reagent": (row.get("reagent") or "").strip(),
        "solvent": (row.get("solvent") or "").strip(),
        "decoded_condition": (row.get("decoded_condition") or "").strip(),
    }


def clean_text(value: object, max_len: int) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def load_existing_results(output_jsonl: Path) -> Dict[int, Dict[str, object]]:
    existing = {}
    if not output_jsonl.exists():
        return existing
    with output_jsonl.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                row_id = int(row["row_id"])
            except Exception:
                continue
            existing[row_id] = row
    return existing


def append_jsonl_rows(output_jsonl: Path, rows: Sequence[Dict[str, object]]) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("a", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")


def sort_output_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda item: (
            -float(item["feasibility_score"]),
            -float(item.get("predicted_yield") or 0.0),
        ),
    )


def output_fieldnames() -> List[str]:
    return [
        "row_id",
        "reaction_smiles",
        "reactant1_smiles",
        "reactant2_smiles",
        "product_smiles",
        "predicted_yield",
        "feasibility_score",
        "probability_level",
        "reaction_family",
        "structural_evidence_summary",
        "analysis_notes",
        "recommended_temperature_c",
        "recommended_conditions",
    ]


def normalize_condition_item(value: object) -> str:
    text = clean_text(value, 500).replace("\u65e0", "none")
    if not text or text.lower() in {"na", "n/a", "none", "not needed", "not required"}:
        return "none"
    return text


def collapse_legacy_condition_slots(row: Dict[str, object], conditions: object) -> str:
    text = clean_text(conditions, 1000).replace("\u65e0", "none")
    if all(label.lower() in text.lower() for label in ("Catalyst:", "Ligand:", "Solvent:")):
        return text
    if not any(field in row for field in ("recommended_catalyst", "recommended_ligand", "recommended_solvent")):
        return text
    return "Catalyst: {0}; Ligand: {1}; Solvent: {2}; Details: {3}".format(
        normalize_condition_item(row.get("recommended_catalyst")),
        normalize_condition_item(row.get("recommended_ligand")),
        normalize_condition_item(row.get("recommended_solvent")),
        text or "none",
    )


def normalize_output_row(row: Dict[str, object]) -> Dict[str, object]:
    fieldnames = output_fieldnames()
    normalized = {field: row.get(field, "") for field in fieldnames}
    normalized["recommended_conditions"] = collapse_legacy_condition_slots(
        row,
        normalized.get("recommended_conditions"),
    )
    return normalized


def write_ranked_csv(output_csv: Path, rows: Sequence[Dict[str, object]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_fieldnames())
        writer.writeheader()
        writer.writerows([normalize_output_row(row) for row in sort_output_rows(rows)])


def write_outputs(output_csv: Path, output_jsonl: Path, rows: Sequence[Dict[str, object]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sort_output_rows(rows)
    write_ranked_csv(output_csv, sorted_rows)
    with output_jsonl.open("w", encoding="utf-8") as outfile:
        for row in sorted_rows:
            outfile.write(json.dumps(normalize_output_row(row), ensure_ascii=False) + "\n")
