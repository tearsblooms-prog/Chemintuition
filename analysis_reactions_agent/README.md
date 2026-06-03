# Reaction Feasibility Agent

This directory contains an LLM-assisted chemical reaction feasibility agent. It ranks candidate reactions by judging whether the written major product is chemically plausible from the supplied reactants and optional condition context.

The agent is intended for screening and triage. It is not a replacement for literature checks, mechanistic validation, atom mapping, or wet-lab decision review.

## Current Pipeline

- Reads candidate reactions from a CSV file.
- Converts each row into a reaction record with reactant SMILES, product SMILES, condition context, and bookkeeping yield.
- Builds local structure evidence with RDKit through `local_reaction_evidence`.
- Sends sanitized reaction batches to an LLM provider.
- Normalizes the LLM response into a score-first output schema.
- Applies local evidence after LLM scoring to cap, raise, or annotate results.
- Writes a ranked CSV, raw JSONL, and progress log.

## Source Layout

```text
analysis_reactions_agent/
|-- run_reaction_feasibility_agent.py  # CLI entrypoint
|-- llm_provider_config.json           # Provider/model/key-file defaults
|-- environment.yml                    # Conda environment with RDKit
|-- requirements.txt                   # Minimal note: prefer Conda env
|-- data/
|   |-- demo_reactions.csv             # Example input CSV
|-- agent_core/
|   |-- app.py                         # CLI args, runtime config, orchestration
|   |-- chemistry.py                   # RDKit parsing/descriptors/similarity
|   |-- config.py                      # Provider and API-key resolution
|   |-- llm.py                         # LLM calls, dry-run mock, response normalization
|   |-- logging_utils.py               # Progress logging and retry diagnostics
|   |-- pipeline.py                    # Batching, retries, workers, snapshots
|   |-- prompts.py                     # System/user prompts
|   |-- records.py                     # CSV/JSONL IO and output normalization
```

Important: current code imports `local_reaction_evidence.py` from the `analysis_reactions_agent/` directory:

```python
from local_reaction_evidence import build_local_evidence, apply_local_evidence
```

That source file must be present before the agent can run. If only `__pycache__/local_reaction_evidence*.pyc` exists, Python will raise `ModuleNotFoundError: No module named 'local_reaction_evidence'`.

## What It Does

- Scores major-product feasibility from `0` to `100`.
- Assigns `probability_level` as `very_high`, `high`, `medium`, or `low`.
- Preserves `predicted_yield` for sorting/bookkeeping but does not send it to the LLM.
- Uses optional `catalyst`, `reagent`, `solvent`, and `decoded_condition` columns as context.
- Produces chemistry notes, recommended temperature, and a single recommended condition string.
- Supports dry-run mode that avoids external API calls.
- Supports resumable JSONL output, heartbeat progress logs, retries, and parallel batch processing.

## What It Does Not Do

- It does not prove a reaction is feasible.
- It does not retrieve literature or patents.
- It does not simulate a mechanism.
- It does not guarantee atom mapping correctness.
- It does not validate experimental safety.
- It should not be used as a standalone gatekeeper for wet-lab decisions.

## Environment

RDKit is a hard dependency, so use the Conda environment from this directory:

```powershell
cd analysis_reactions_agent
conda env create -f environment.yml
conda activate yieldnet-reaction-agent
```

The repository root may use older Python environments. This agent environment is pinned to Python 3.10 in `environment.yml`.

Quick import check:

```powershell
python -c "from rdkit import Chem; from agent_core.chemistry import rdkit_is_available; print('rdkit_ok=', rdkit_is_available())"
```

Source completeness check:

```powershell
Test-Path .\local_reaction_evidence.py
```

## API Configuration

Provider defaults are defined in `agent_core/config.py`:

- `gemini`: default provider, model `gemini-2.5-flash`
- `deepseek`: alternative provider, model `deepseek-v4-flash`

The runtime resolves configuration in this order:

- CLI arguments such as `--provider`, `--model`, `--api-url`, and `--api-key`
- `llm_provider_config.json`
- provider key files such as `gemini_api_key.txt` or `deepseek_api_key.txt`
- environment variables `GEMINI_API_KEY` or `DEEPSEEK_API_KEY`

Environment variables are preferred for secrets:

```powershell
$env:GEMINI_API_KEY="your_key"
$env:DEEPSEEK_API_KEY="your_key"
```

Use `--dry-run` when you want to test local IO and batching without an API key.

## Input Schema

Required columns:

- `reactant1_smiles`
- `reactant2_smiles`
- `product_smiles`
- `predicted_yield`

Optional context columns:

- `catalyst`
- `reagent`
- `solvent`
- `decoded_condition`

Other columns, such as `condition_fingerprint`, can remain in the CSV. They are ignored by the current record builder.

Default input:

```text
data/demo_reactions.csv
```

## Output Schema

The ranked CSV and final JSONL contain these fields:

- `row_id`
- `reaction_smiles`
- `reactant1_smiles`
- `reactant2_smiles`
- `product_smiles`
- `predicted_yield`
- `feasibility_score`
- `probability_level`
- `reaction_family`
- `structural_evidence_summary`
- `analysis_notes`
- `recommended_temperature_c`
- `recommended_conditions`

Notes:

- `keep` is no longer returned. The current workflow is score-first.
- `recommended_conditions` is one English string formatted as `Catalyst: ...; Ligand: ...; Solvent: ...; Details: ...`.
- Missing or unjustified condition items are normalized to `none`.
- Intermediate normalized rows may include `risk_flags` and `likely_known_reaction`, but `records.output_fieldnames()` does not write them to the final CSV/JSONL schema.

## Default Files

- Input CSV: `data/demo_reactions.csv`
- Ranked CSV: `data/demo_reactions_ranked.csv`
- Raw JSONL: `data/demo_reactions_raw.jsonl`
- Progress log: `data/demo_reactions_feasibility_progress.txt`
- Provider config: `llm_provider_config.json`

## Running

Run all commands from `analysis_reactions_agent/`.

Dry run without external API calls:

```powershell
conda run -n yieldnet-reaction-agent python run_reaction_feasibility_agent.py --dry-run --no-resume --limit 20
```

Run with the configured LLM provider:

```powershell
conda run -n yieldnet-reaction-agent python run_reaction_feasibility_agent.py --no-resume --limit 20 --batch-size 5 --max-workers 2
```

Run the full default file:

```powershell
conda run -n yieldnet-reaction-agent python run_reaction_feasibility_agent.py --batch-size 8 --max-workers 4
```

Use a custom input/output set:

```powershell
conda run -n yieldnet-reaction-agent python run_reaction_feasibility_agent.py `
  --input data/my_reactions.csv `
  --output-csv data/my_reactions_ranked.csv `
  --output-jsonl data/my_reactions_raw.jsonl `
  --progress-log data/my_reactions_progress.txt
```

## CLI Options

- `--input`: input CSV path.
- `--output-csv`: ranked output CSV path.
- `--output-jsonl`: raw/resume JSONL path.
- `--progress-log`: progress log path.
- `--config`: provider config JSON path.
- `--provider`: provider name, currently `gemini` or `deepseek`.
- `--api-key`: API key override.
- `--api-url`: provider endpoint override.
- `--model`: model override.
- `--batch-size`: reactions per LLM request.
- `--max-workers`: number of parallel batch workers.
- `--keep-threshold`: local-evidence threshold passed into normalization.
- `--limit`: maximum number of rows to process.
- `--offset`: zero-based row offset.
- `--temperature`: LLM sampling temperature.
- `--top-p`: LLM nucleus sampling value.
- `--max-retries`: retry count per batch.
- `--timeout`: HTTP timeout in seconds.
- `--heartbeat-seconds`: interval for active-batch heartbeat logs.
- `--csv-snapshot-seconds`: interval for partial CSV snapshots; use `0` to disable snapshots.
- `--dry-run`: use the local mock analyzer instead of calling an LLM.
- `--no-resume`: ignore existing JSONL rows and recompute selected rows.
- `--debug-notes`: write local-evidence debug notes to a file.

## Resume And Progress

- Existing rows in `--output-jsonl` are reused unless `--no-resume` is passed.
- Completed batches append rows to JSONL immediately.
- The final CSV is sorted by descending `feasibility_score`, then descending `predicted_yield`.
- Heartbeat logs record active batches, attempts, row ranges, retries, and completion counts.
- Partial CSV snapshots are written during long runs when `--csv-snapshot-seconds` is greater than `0`.

## LLM Payload

The LLM receives only sanitized fields:

- `row_id`
- `reaction_smiles`
- `reactant1_smiles`
- `reactant2_smiles`
- `product_smiles`
- `catalyst`
- `reagent`
- `solvent`
- `decoded_condition`
- `structural_evidence_summary`
- `inference_rules`

`predicted_yield` is intentionally excluded from the LLM payload to avoid using the upstream yield model as a prior for feasibility scoring.

## Practical Guidance

- Treat `feasibility_score` as a ranking signal, not proof.
- Manually inspect low-score false negatives and high-score false positives.
- Use `structural_evidence_summary` and `analysis_notes` to understand why a score changed.
- Use `--dry-run` and `--limit` before spending API calls on a large file.
- If literature-aware novelty or known-reaction checks are needed, add a retrieval module rather than relying on the LLM alone.

## Known Checks

- `local_reaction_evidence.py` must exist in this directory.
- `verify_chem_stack.py` is not present in the current tree, so older verification commands that call it are obsolete.
- `setup_conda_env.bat` still references `verify_chem_stack.py`; update that script before using it as an automated setup helper.
- `.env.example` is a reference file only; the current code does not auto-load `.env` files or read `GEMINI_MODEL`. Use `--model` or `llm_provider_config.json` for model overrides.
