from __future__ import annotations

import json
import urllib.request
from typing import Dict, List, Optional, Sequence

from local_reaction_evidence import apply_local_evidence

from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .records import clean_text, parse_float

NO_CONDITION = "none"


class LLMClient(object):
    def __init__(
        self,
        provider: str,
        api_key: str,
        api_url: str,
        model: str,
        timeout: int,
        temperature: float,
        top_p: float,
    ):
        self.provider = provider
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p

    def analyze_batch(self, batch: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        payload = json.dumps(serialize_batch_for_llm(batch), ensure_ascii=False, indent=2)
        request_body = self.build_request_body(payload)
        request = urllib.request.Request(
            self.api_url.format(model=self.model),
            data=json.dumps(request_body).encode("utf-8"),
            headers=self.build_headers(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        content = self.extract_response_text(data)
        if not content:
            raise ValueError("Model returned empty content")
        parsed = json.loads(content)
        return normalize_model_response(parsed)

    def build_request_body(self, payload: str) -> Dict[str, object]:
        if self.provider == "gemini":
            return {
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": USER_PROMPT_TEMPLATE.format(payload=payload)}]}],
                "generationConfig": {
                    "temperature": self.temperature,
                    "topP": self.top_p,
                    "responseMimeType": "application/json",
                },
            }
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(payload=payload)},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "response_format": {"type": "json_object"},
        }

    def build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.provider == "gemini":
            headers["x-goog-api-key"] = self.api_key
        else:
            headers["Authorization"] = "Bearer {0}".format(self.api_key)
        return headers

    def extract_response_text(self, data: Dict[str, object]) -> str:
        if self.provider == "gemini":
            candidates = data.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("Gemini response does not contain candidates")
            content = candidates[0].get("content", {})
            parts = content.get("parts", []) if isinstance(content, dict) else []
            text_parts = [str(part.get("text") or "") for part in parts if isinstance(part, dict)]
            return "".join(text_parts).strip()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("DeepSeek response does not contain choices")
        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            raise ValueError("DeepSeek response message is invalid")
        return str(message.get("content") or "").strip()


def serialize_batch_for_llm(batch: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    allowed_fields = (
        "row_id",
        "reaction_smiles",
        "reactant1_smiles",
        "reactant2_smiles",
        "product_smiles",
        "catalyst",
        "reagent",
        "solvent",
        "decoded_condition",
        "structural_evidence_summary",
        "inference_rules",
    )
    sanitized_batch = []
    for item in batch:
        sanitized_batch.append({field: item.get(field, "") for field in allowed_fields if field in item})
    return sanitized_batch


def mock_analyze_batch(batch: Sequence[Dict[str, object]], keep_threshold: float) -> List[Dict[str, object]]:
    results = []
    for item in batch:
        product = str(item.get("product_smiles") or "")
        reactants = ".".join([str(item.get("reactant1_smiles") or ""), str(item.get("reactant2_smiles") or "")]).strip(".")
        evidence_summary = str(item.get("structural_evidence_summary") or "").lower()
        score = 55.0
        if not product or not reactants:
            score = 15.0
        elif "major core-framework shift" in evidence_summary or "retains little of the reactant scaffold" in evidence_summary:
            score = 34.0
        elif "very low for the written reactant/product pair" in evidence_summary:
            score = 38.0
        elif "meaningful whole-structure continuity" in evidence_summary and "substantial product scaffold" in evidence_summary:
            score = 78.0
        elif "substantial product scaffold" in evidence_summary:
            score = 68.0
        results.append(
            {
                "row_id": item["row_id"],
                "feasibility_score": round(score, 2),
                "probability_level": infer_probability_level(score),
                "reasoning": "Dry-run mock result based on structural evidence only, without using upstream predicted yield.",
                "recommended_temperature_c": 25.0 if score < 70 else 60.0,
                "recommended_conditions": "Catalyst: none; Ligand: none; Solvent: none; Details: Need LLM analysis for a chemistry-grounded recommendation.",
                "likely_known_reaction": False,
                "risk_flags": ["dry_run_no_llm"],
            }
        )
    return results


def normalize_model_response(parsed: object) -> List[Dict[str, object]]:
    if isinstance(parsed, dict):
        reactions = parsed.get("reactions")
    elif isinstance(parsed, list):
        reactions = parsed
    else:
        raise ValueError("Model response must be a JSON object or JSON array")
    if not isinstance(reactions, list):
        raise ValueError("Model response does not contain a reactions list")

    flattened: List[Dict[str, object]] = []

    def visit(item: object) -> None:
        if isinstance(item, dict):
            flattened.append(item)
            return
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        raise ValueError("Model response contains a non-object reaction item")

    for item in reactions:
        visit(item)
    if not flattened:
        raise ValueError("Model response contains no reaction objects")
    return flattened


def infer_probability_level(score: float) -> str:
    if score >= 85:
        return "very_high"
    if score >= 70:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


def normalize_condition_item(value: object, fallback: object = "") -> str:
    for candidate in (value, fallback):
        text = clean_text(candidate, 500).replace("\u65e0", NO_CONDITION)
        if text:
            if text.lower() in {"na", "n/a", "none", "not needed", "not required"}:
                return NO_CONDITION
            return text
    return NO_CONDITION


def normalize_recommended_conditions(reaction: Dict[str, object], source_row: Dict[str, str]) -> str:
    conditions = clean_text(reaction.get("recommended_conditions"), 1000).replace("\u65e0", NO_CONDITION)
    labels = ("Catalyst:", "Ligand:", "Solvent:")
    if all(label.lower() in conditions.lower() for label in labels):
        return conditions

    catalyst = normalize_condition_item(reaction.get("recommended_catalyst"), source_row.get("catalyst"))
    ligand = normalize_condition_item(reaction.get("recommended_ligand"))
    solvent = normalize_condition_item(reaction.get("recommended_solvent"), source_row.get("solvent"))
    details = conditions or NO_CONDITION
    return "Catalyst: {0}; Ligand: {1}; Solvent: {2}; Details: {3}".format(
        catalyst,
        ligand,
        solvent,
        details,
    )


def normalize_analysis(
    reaction: Dict[str, object],
    source_row: Dict[str, str],
    keep_threshold: float,
    local_evidence: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    score = float(reaction.get("feasibility_score") or 0.0)
    risk_flags = reaction.get("risk_flags")
    if not isinstance(risk_flags, list):
        risk_flags = []
    probability_level = str(reaction.get("probability_level") or infer_probability_level(score))
    recommended_temperature = reaction.get("recommended_temperature_c")
    try:
        recommended_temperature = float(recommended_temperature)
    except (TypeError, ValueError):
        recommended_temperature = 25.0
    predicted_yield = parse_float(source_row.get("predicted_yield"), default=0.0)
    normalized = {
        "row_id": int(reaction["row_id"]),
        "reaction_smiles": ">".join(
            [
                ".".join([(source_row.get("reactant1_smiles") or "").strip(), (source_row.get("reactant2_smiles") or "").strip()]).strip("."),
                "",
                (source_row.get("product_smiles") or "").strip(),
            ]
        ),
        "reactant1_smiles": source_row.get("reactant1_smiles", ""),
        "reactant2_smiles": source_row.get("reactant2_smiles", ""),
        "product_smiles": source_row.get("product_smiles", ""),
        "predicted_yield": predicted_yield,
        "feasibility_score": round(score, 2),
        "probability_level": probability_level,
        "analysis_notes": clean_text(reaction.get("reasoning"), 2000),
        "recommended_temperature_c": recommended_temperature,
        "recommended_conditions": normalize_recommended_conditions(reaction, source_row),
        "likely_known_reaction": bool(reaction.get("likely_known_reaction", False)),
        "risk_flags": "|".join([str(flag) for flag in risk_flags]),
    }
    return apply_local_evidence(normalized, local_evidence, keep_threshold)
