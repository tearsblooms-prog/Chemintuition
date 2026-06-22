from __future__ import annotations

from typing import Dict, Optional, Sequence

from agent_core.chemistry import build_reaction_profile


NEGATIVE_MARKERS = (
    "impossible",
    "implausible",
    "incompatible",
    "contradictory",
    "mismatch",
    "cannot be formed",
    "not plausible",
)


def combined_reactants(row: Dict[str, str]) -> str:
    return ".".join(
        value.strip()
        for value in ((row.get("reactant1_smiles") or ""), (row.get("reactant2_smiles") or ""))
        if value.strip()
    )


def detect_family(row: Dict[str, str], profile: Dict[str, object]) -> str:
    reactants = combined_reactants(row)
    product = (row.get("product_smiles") or "").strip()
    reagent = (row.get("reagent") or "").strip()
    motifs = profile.get("motifs") if isinstance(profile.get("motifs"), dict) else {}

    has_alkene = bool(motifs.get("alkene")) or "=" in reactants
    has_boron_in_reactants = bool(motifs.get("boronic_acid_or_ester")) or "B" in reactants
    has_bpin_in_product = bool(motifs.get("product_boronate")) or "B(" in product or "B1O" in product
    has_aryl_halide = bool(motifs.get("aryl_halide"))
    has_disulfide = bool(motifs.get("disulfide")) or "SS" in reactants or "CSSC" in reactants
    has_phosphorus = bool(motifs.get("phosphorus")) or "P" in reactants
    has_benzyl_ether = bool(motifs.get("benzyl_ether")) or ("OCc" in reactants and "c1" in reactants)
    keeps_benzyl_ether = bool(motifs.get("product_benzyl_ether")) or ("OCc" in product and "c1" in product)
    has_carboxyl = bool(motifs.get("carboxyl")) or "C(=O)O" in reactants or "C(=O)[O-]" in reactants
    has_acyl_halide = bool(motifs.get("acyl_halide")) or "C(=O)Cl" in reactants or "C(=O)F" in reactants
    product_has_anhydride = "OC(=O)" in product and product.count("C(=O)") >= 2
    descriptor_delta = profile.get("descriptor_delta") if isinstance(profile.get("descriptor_delta"), dict) else {}
    delta_rings = float(descriptor_delta.get("rings") or 0.0)
    product_keeps_aryl_halide = bool(motifs.get("product_aryl_halide"))

    if has_aryl_halide and has_boron_in_reactants and not has_bpin_in_product:
        return "c_c_cross_coupling"
    if has_aryl_halide and not product_keeps_aryl_halide and "N" in product:
        return "c_n_coupling"
    if has_aryl_halide and not product_keeps_aryl_halide and "O" in product:
        return "c_o_coupling"
    if has_alkene and has_bpin_in_product:
        return "borylation"
    if has_disulfide and has_phosphorus and ("SP(" in product or "P(=S)" in product):
        return "heteroatom_functionalization"
    if has_benzyl_ether and not keeps_benzyl_ether and "O" in product:
        return "protection_deprotection"
    if has_carboxyl and "C(=O)" in product:
        return "acylation"
    if has_acyl_halide and product_has_anhydride:
        return "substitution"
    if "NaH" in reagent and has_alkene and has_bpin_in_product:
        return "borylation"
    if reactants.count("=") > product.count("=") and "H" not in product:
        return "hydrogenation_reduction"
    if ("BrC" in reactants or "ClC" in reactants) and ("=O" in product):
        return "oxidation"
    if delta_rings >= 1.0:
        return "condensation_cyclization"
    return "general"


def infer_probability_level(score: float) -> str:
    if score >= 85.0:
        return "very_high"
    if score >= 70.0:
        return "high"
    if score >= 50.0:
        return "medium"
    return "low"


def analyze_structure_consistency(row: Dict[str, str]) -> Dict[str, object]:
    reactants = combined_reactants(row)
    product = (row.get("product_smiles") or "").strip()
    profile = build_reaction_profile(row)
    family = detect_family(row, profile)
    risk_flags = []
    evidence_lines = []
    score_floor = 58.0
    score_ceiling = 100.0

    if not product or not reactants:
        return {
            "family": family,
            "score_floor": 15.0,
            "keep_floor": False,
            "score_ceiling": 25.0,
            "risk_flags": ["missing_structure"],
            "summary": "Reaction is missing reactant or product structure.",
            "override_reasoning": "",
            "recommended_conditions": "",
            "profile_summary": "",
        }

    profile_summary = str(profile.get("summary") or "").strip()
    if profile_summary:
        evidence_lines.append("Cheminformatics: {0}".format(profile_summary))

    invalid_smiles = list(profile.get("invalid_smiles") or [])
    if invalid_smiles:
        return {
            "family": family,
            "score_floor": 18.0,
            "keep_floor": False,
            "score_ceiling": 25.0,
            "risk_flags": ["invalid_smiles"],
            "summary": "SMILES parsing failed for one or more structures: {0}".format("; ".join(invalid_smiles)),
            "override_reasoning": "",
            "recommended_conditions": "",
            "profile_summary": profile_summary,
        }

    shared_atoms = float(profile.get("shared_atoms") or 0.0)
    product_atom_coverage = float(profile.get("product_atom_coverage") or 0.0)
    reactant_atom_coverage = float(profile.get("reactant_atom_coverage") or 0.0)
    fingerprint_tanimoto = float(profile.get("fingerprint_tanimoto") or 0.0)
    murcko_tanimoto = float(profile.get("murcko_tanimoto") or 0.0)
    murcko_shared_atoms = float(profile.get("murcko_shared_atoms") or 0.0)
    murcko_exact_match = bool(profile.get("murcko_exact_match"))
    product_new_elements = list(profile.get("product_new_elements") or [])
    reactant_contributions = profile.get("reactant_contributions") if isinstance(profile.get("reactant_contributions"), list) else []
    descriptor_delta = profile.get("descriptor_delta") if isinstance(profile.get("descriptor_delta"), dict) else {}
    delta_logp = float(descriptor_delta.get("logp") or 0.0)
    delta_tpsa = float(descriptor_delta.get("tpsa") or 0.0)
    delta_hba = float(descriptor_delta.get("hba") or 0.0)
    delta_hbd = float(descriptor_delta.get("hbd") or 0.0)
    motifs = profile.get("motifs") if isinstance(profile.get("motifs"), dict) else {}

    strongest_contribution = 0.0
    weakest_contribution = 1.0
    if reactant_contributions:
        strongest_contribution = max(float(item.get("product_atom_coverage") or 0.0) for item in reactant_contributions)
        weakest_contribution = min(float(item.get("product_atom_coverage") or 0.0) for item in reactant_contributions)
    has_boron_reactant = bool(motifs.get("boronic_acid_or_ester"))
    has_silicon_reactant = bool(motifs.get("silicon"))
    has_benzylic_halide = bool(motifs.get("benzylic_halide"))
    product_is_carbonyl = bool(motifs.get("product_aldehyde")) or bool(motifs.get("product_ketone"))

    if shared_atoms <= 0 and profile.get("rdkit_available"):
        score_floor = 22.0
        risk_flags.append("core_scaffold_break")
        evidence_lines.append("RDKit could not find a meaningful shared scaffold between reactants and product.")
    elif product_atom_coverage < 0.25:
        score_floor = min(score_floor, 38.0)
        risk_flags.append("core_scaffold_break")
        evidence_lines.append("Only a small fraction of the product heavy-atom scaffold overlaps with the written reactants.")
    elif product_atom_coverage >= 0.55:
        evidence_lines.append("A substantial product scaffold is retained from the reactants.")

    if fingerprint_tanimoto >= 0.45:
        evidence_lines.append("Morgan fingerprint similarity indicates meaningful whole-structure continuity.")
    elif fingerprint_tanimoto <= 0.12:
        score_floor = min(score_floor, 36.0)
        risk_flags.append("core_scaffold_break")
        evidence_lines.append("Morgan fingerprint similarity is very low for the written reactant/product pair.")

    if murcko_exact_match or murcko_tanimoto >= 0.55 or murcko_shared_atoms >= 8:
        evidence_lines.append("Bemis-Murcko scaffold analysis supports preservation of the core framework.")
    elif product_atom_coverage < 0.35 and murcko_tanimoto <= 0.10:
        score_floor = min(score_floor, 34.0)
        risk_flags.append("core_scaffold_break")
        evidence_lines.append("Bemis-Murcko scaffold analysis suggests a major core-framework shift.")

    if product_new_elements:
        score_floor = min(score_floor, 35.0)
        risk_flags.append("new_element_introduced")
        evidence_lines.append(
            "Product introduces element types not present in the written reactants: {0}.".format(", ".join(product_new_elements))
        )

    if abs(delta_tpsa) >= 45.0 or abs(delta_logp) >= 3.5:
        risk_flags.append("redox_mismatch")
        evidence_lines.append(
            "Physicochemical descriptors change sharply across the reaction (delta_tpsa={0:.1f}, delta_logp={1:.1f}).".format(
                delta_tpsa,
                delta_logp,
            )
        )
        if product_atom_coverage < 0.35:
            score_floor = min(score_floor, 42.0)

    if abs(delta_hba) >= 4.0 or abs(delta_hbd) >= 3.0:
        risk_flags.append("chemoselectivity_risk")
        evidence_lines.append(
            "Hydrogen-bond donor/acceptor counts shift substantially (delta_hba={0:.0f}, delta_hbd={1:.0f}).".format(
                delta_hba,
                delta_hbd,
            )
        )

    if family == "c_c_cross_coupling":
        score_floor = max(score_floor, 74.0 if product_atom_coverage >= 0.45 or murcko_tanimoto >= 0.45 else 66.0)
        evidence_lines.append("Pd-catalyzed halide/boron cross-coupling pattern detected.")
        if product_atom_coverage < 0.40:
            risk_flags.append("core_scaffold_break")
        risk_flags.append("missing_partner_possible")
    elif family == "c_n_coupling":
        score_floor = max(score_floor, 66.0 if product_atom_coverage >= 0.35 else 60.0)
        evidence_lines.append("A C-N bond-forming coupling pattern is plausible from the written structures.")
        risk_flags.append("chemoselectivity_risk")
    elif family == "c_o_coupling":
        score_floor = max(score_floor, 64.0 if product_atom_coverage >= 0.35 else 58.0)
        evidence_lines.append("A C-O bond-forming coupling pattern is plausible from the written structures.")
        risk_flags.append("chemoselectivity_risk")
    elif family == "borylation":
        score_floor = max(score_floor, 70.0 if reactant_atom_coverage >= 0.30 else 64.0)
        evidence_lines.append("Alkene-to-boronate family detected from reactant and product motifs.")
        if "B" not in reactants:
            risk_flags.append("missing_partner_possible")
            evidence_lines.append("Missing explicit boron source is treated as incomplete bookkeeping rather than a hard contradiction.")
        risk_flags.append("regioselectivity_risk")
    elif family == "heteroatom_functionalization":
        strong_s_p_transfer_support = fingerprint_tanimoto >= 0.50 and (
            product_atom_coverage >= 0.48 or murcko_shared_atoms >= 14
        )
        score_floor = max(score_floor, 76.0 if strong_s_p_transfer_support else 70.0)
        evidence_lines.append("A heteroatom transfer or heteroatom-focused functionalization pattern is structurally coherent.")
        risk_flags.append("chemoselectivity_risk")
    elif family == "protection_deprotection":
        score_floor = max(score_floor, 74.0)
        evidence_lines.append("Protecting-group removal pattern detected; omitted benzyl-derived side products are expected.")
        risk_flags.append("chemoselectivity_risk")
    elif family == "acylation":
        score_floor = max(score_floor, 64.0)
        evidence_lines.append("Acyl donor plus oxygenated substrate pattern is consistent with acylation chemistry.")
        risk_flags.append("regioselectivity_risk")
    elif family == "substitution":
        score_floor = max(score_floor, 56.0)
        evidence_lines.append("A substitution-like transformation is plausible, but partner bookkeeping may be incomplete.")
        risk_flags.append("missing_partner_possible")
    elif family == "oxidation":
        score_floor = max(score_floor, 52.0)
        evidence_lines.append("An oxidation-like functional-group change is suggested by the written structures.")
        risk_flags.append("redox_mismatch")
    elif family == "hydrogenation_reduction":
        score_floor = max(score_floor, 52.0)
        evidence_lines.append("A reduction-like change in unsaturation is suggested by the written structures.")
        risk_flags.append("redox_mismatch")
    elif family == "condensation_cyclization":
        score_floor = max(score_floor, 58.0)
        evidence_lines.append("A condensation or cyclization pattern is plausible from the scaffold change.")
        risk_flags.append("regioselectivity_risk")
    else:
        if product_atom_coverage >= 0.58 and fingerprint_tanimoto >= 0.33 and (murcko_tanimoto >= 0.30 or murcko_shared_atoms >= 8):
            score_floor = max(score_floor, 60.0)
            evidence_lines.append("No named family was matched, but multiple orthogonal similarity signals support a plausible transformation.")
        elif product_atom_coverage < 0.20:
            score_floor = min(score_floor, 40.0)
            risk_flags.append("core_scaffold_break")
            evidence_lines.append("No strong hand-coded family was matched and the product retains little of the reactant scaffold.")
        else:
            evidence_lines.append("No strong hand-coded reaction family detected; leave the final judgment to the LLM with a neutral prior.")

    if family == "general" and reactant_contributions and len(reactant_contributions) >= 2 and weakest_contribution <= 0.15:
        score_ceiling = min(score_ceiling, 46.0)
        risk_flags.append("spectator_reactant")
        evidence_lines.append("One written reactant contributes almost none of the product scaffold, but no supported reagent role was detected.")

    if family == "general" and has_boron_reactant and not ("B" in product):
        score_ceiling = min(score_ceiling, 44.0)
        risk_flags.append("missing_partner_possible")
        evidence_lines.append("A boron-bearing reactant disappears without a supported cross-coupling or borylation pattern.")

    if family == "general" and has_silicon_reactant and "Si" not in product:
        score_ceiling = min(score_ceiling, 40.0)
        risk_flags.append("missing_partner_possible")
        evidence_lines.append("A silicon reagent disappears without a supported silylation or deprotection pattern.")

    if family == "general" and murcko_exact_match and fingerprint_tanimoto <= 0.24 and strongest_contribution >= 0.60:
        score_ceiling = min(score_ceiling, 42.0)
        risk_flags.append("core_scaffold_break")
        evidence_lines.append("The apparent match is driven mainly by a trivial shared core scaffold rather than whole-molecule continuity.")

    if family == "general" and has_benzylic_halide and product_is_carbonyl and fingerprint_tanimoto <= 0.24:
        score_ceiling = min(score_ceiling, 38.0)
        risk_flags.append("redox_mismatch")
        evidence_lines.append("The product requires an unsupported benzylic oxidation or carbonyl rewrite not grounded by the written reactants.")

    recommended_conditions = {
        "c_c_cross_coupling": "Catalyst: Pd catalyst; Ligand: biaryl phosphine ligand; Solvent: 1,4-dioxane/H2O; Details: Add a base under standard cross-coupling conditions.",
        "c_n_coupling": "Catalyst: Pd catalyst; Ligand: biaryl phosphine ligand; Solvent: 1,4-dioxane; Details: Match base and heating to the aryl halide and amine.",
        "c_o_coupling": "Catalyst: Cu catalyst; Ligand: N,N-donor ligand; Solvent: DMSO; Details: Match the base to the leaving group and oxygen nucleophile.",
        "borylation": "Catalyst: none; Ligand: none; Solvent: THF; Details: Use a boron source such as HBpin or B2pin2 with a compatible promoter.",
        "heteroatom_functionalization": "Catalyst: none; Ligand: none; Solvent: THF; Details: Choose heteroatom-transfer conditions that preserve the core scaffold.",
        "protection_deprotection": "Catalyst: Pd/C; Ligand: none; Solvent: MeOH; Details: Use hydrogenolysis only when the protecting group and substrate tolerate it.",
        "acylation": "Catalyst: none; Ligand: none; Solvent: CH2Cl2; Details: Use an acyl donor and activation mode matched to the substrate.",
        "substitution": "Catalyst: none; Ligand: none; Solvent: DMF; Details: Match base and solvent polarity to the leaving group and nucleophile.",
        "oxidation": "Catalyst: none; Ligand: none; Solvent: CH2Cl2; Details: Use an oxidant aligned with the substrate oxidation state and functional-group tolerance.",
        "hydrogenation_reduction": "Catalyst: Pd/C; Ligand: none; Solvent: EtOH; Details: Use catalytic hydrogenation only for compatible reducible motifs.",
        "condensation_cyclization": "Catalyst: none; Ligand: none; Solvent: toluene; Details: Use activating or water-removal conditions matched to the ring-forming motif.",
    }.get(family, "")

    override_reasoning = {
        "c_c_cross_coupling": "Local structural evidence supports a plausible C-C cross-coupling assignment.",
        "c_n_coupling": "Local structural evidence supports a plausible C-N bond-forming assignment.",
        "c_o_coupling": "Local structural evidence supports a plausible C-O bond-forming assignment.",
        "borylation": "Local structural evidence supports a plausible borylation assignment.",
        "heteroatom_functionalization": "Local structural evidence supports a plausible heteroatom functionalization assignment.",
        "protection_deprotection": "Local structural evidence supports a plausible protection or deprotection assignment.",
        "acylation": "Local structural evidence supports a plausible acylation assignment.",
        "substitution": "Local structural evidence supports a plausible substitution assignment.",
        "oxidation": "Local structural evidence supports a plausible oxidation assignment.",
        "hydrogenation_reduction": "Local structural evidence supports a plausible reduction assignment.",
        "condensation_cyclization": "Local structural evidence supports a plausible condensation or cyclization assignment.",
    }.get(family, "")

    return {
        "family": family,
        "score_floor": score_floor,
        "score_ceiling": score_ceiling,
        "keep_floor": False,
        "risk_flags": risk_flags,
        "summary": " ".join(line.strip() for line in evidence_lines if line.strip()).strip(),
        "override_reasoning": override_reasoning,
        "recommended_conditions": recommended_conditions,
        "profile_summary": profile_summary,
    }


def build_local_evidence(
    records: Sequence[Dict[str, str]],
    corpus_path: Optional[object] = None,
    top_k: int = 0,
    max_rows: int = 0,
) -> Dict[int, Dict[str, object]]:
    evidence_map: Dict[int, Dict[str, object]] = {}
    for record in records:
        evidence_map[int(record["row_id"])] = analyze_structure_consistency(record)
    return evidence_map


def apply_local_evidence(result: Dict[str, object], evidence: Optional[Dict[str, object]], keep_threshold: float) -> Dict[str, object]:
    if not evidence:
        return result

    original_score = float(result.get("feasibility_score") or 0.0)
    score_floor = float(evidence.get("score_floor") or 0.0)
    score_ceiling = float(evidence.get("score_ceiling") or 100.0)
    final_score = min(max(original_score, score_floor), score_ceiling)
    score_is_high = final_score >= keep_threshold
    result["feasibility_score"] = round(final_score, 2)
    result["reaction_family"] = str(evidence.get("family") or "general")

    local_summary = str(evidence.get("summary") or "").strip()
    result["structural_evidence_summary"] = local_summary
    if local_summary:
        existing = str(result.get("analysis_notes") or "").strip()
        result["analysis_notes"] = "{0} Structural evidence: {1}".format(existing, local_summary).strip()

    risk_flags = [flag for flag in str(result.get("risk_flags") or "").split("|") if flag]
    for flag in evidence.get("risk_flags") or []:
        flag_text = str(flag)
        if flag_text not in risk_flags:
            risk_flags.append(flag_text)
    result["risk_flags"] = "|".join(risk_flags)
    result["probability_level"] = infer_probability_level(final_score)

    recommended_conditions = str(result.get("recommended_conditions") or "").strip()
    if score_is_high and not recommended_conditions:
        fallback_conditions = str(evidence.get("recommended_conditions") or "").strip()
        if fallback_conditions:
            result["recommended_conditions"] = fallback_conditions

    analysis_notes = str(result.get("analysis_notes") or "").lower()
    if score_is_high and any(marker in analysis_notes for marker in NEGATIVE_MARKERS):
        override_reasoning = str(evidence.get("override_reasoning") or "").strip()
        if override_reasoning and final_score >= 68.0:
            result["analysis_notes"] = "{0} Structural evidence: {1}".format(override_reasoning, local_summary).strip()

    return result
