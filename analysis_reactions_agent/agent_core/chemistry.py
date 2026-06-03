from __future__ import annotations

from typing import Dict, List, Sequence

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFMCS, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


def rdkit_is_available() -> bool:
    return (
        Chem is not None
        and DataStructs is not None
        and Crippen is not None
        and Lipinski is not None
        and rdFMCS is not None
        and rdMolDescriptors is not None
        and GetMorganGenerator is not None
        and MurckoScaffold is not None
    )


def split_smiles(smiles: str) -> List[str]:
    return [part.strip() for part in (smiles or "").split(".") if part.strip()]


def parse_smiles(smiles: str):
    if not rdkit_is_available() or not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def combine_mols(smiles_values: Sequence[str]):
    if not rdkit_is_available():
        return None
    mols = [parse_smiles(smiles) for smiles in smiles_values if smiles]
    mols = [mol for mol in mols if mol is not None]
    if not mols:
        return None
    combined = mols[0]
    for mol in mols[1:]:
        combined = Chem.CombineMols(combined, mol)
    combined = Chem.Mol(combined)
    Chem.SanitizeMol(combined)
    return combined


def canonicalize_smiles(smiles: str) -> str:
    mol = parse_smiles(smiles)
    if mol is None or not rdkit_is_available():
        return (smiles or "").strip()
    return Chem.MolToSmiles(mol, canonical=True)


def mol_formula(mol) -> str:
    if mol is None or not rdkit_is_available():
        return ""
    return rdMolDescriptors.CalcMolFormula(mol)


def heavy_atom_count(mol) -> int:
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def ring_count(mol) -> int:
    if mol is None or not rdkit_is_available():
        return 0
    return int(rdMolDescriptors.CalcNumRings(mol))


def hetero_atom_count(mol) -> int:
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))


def exact_mol_wt(mol) -> float:
    if mol is None or Descriptors is None:
        return 0.0
    return float(Descriptors.ExactMolWt(mol))


def tpsa(mol) -> float:
    if mol is None or rdMolDescriptors is None:
        return 0.0
    return float(rdMolDescriptors.CalcTPSA(mol))


def logp(mol) -> float:
    if mol is None or Crippen is None:
        return 0.0
    return float(Crippen.MolLogP(mol))


def hba_count(mol) -> int:
    if mol is None or Lipinski is None:
        return 0
    return int(Lipinski.NumHAcceptors(mol))


def hbd_count(mol) -> int:
    if mol is None or Lipinski is None:
        return 0
    return int(Lipinski.NumHDonors(mol))


def rotatable_bonds(mol) -> int:
    if mol is None or Lipinski is None:
        return 0
    return int(Lipinski.NumRotatableBonds(mol))


def fraction_csp3(mol) -> float:
    if mol is None or rdMolDescriptors is None:
        return 0.0
    return float(rdMolDescriptors.CalcFractionCSP3(mol))


def aromatic_ring_count(mol) -> int:
    if mol is None or rdMolDescriptors is None:
        return 0
    return int(rdMolDescriptors.CalcNumAromaticRings(mol))


def element_set(mol) -> List[str]:
    if mol is None:
        return []
    return sorted({atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1})


def has_substructure(mol, smarts: str) -> bool:
    if mol is None or not rdkit_is_available():
        return False
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return False
    return mol.HasSubstructMatch(pattern)


def descriptor_snapshot(mol) -> Dict[str, float]:
    if mol is None:
        return {}
    return {
        "heavy_atoms": float(heavy_atom_count(mol)),
        "rings": float(ring_count(mol)),
        "hetero_atoms": float(hetero_atom_count(mol)),
        "exact_mw": round(exact_mol_wt(mol), 3),
        "tpsa": round(tpsa(mol), 3),
        "logp": round(logp(mol), 3),
        "hba": float(hba_count(mol)),
        "hbd": float(hbd_count(mol)),
        "rotatable_bonds": float(rotatable_bonds(mol)),
        "fraction_csp3": round(fraction_csp3(mol), 3),
        "aromatic_rings": float(aromatic_ring_count(mol)),
    }


def descriptor_delta(reactant_data: Dict[str, float], product_data: Dict[str, float]) -> Dict[str, float]:
    if not reactant_data or not product_data:
        return {}
    keys = set(reactant_data) & set(product_data)
    return {key: round(float(product_data[key]) - float(reactant_data[key]), 3) for key in sorted(keys)}


def morgan_fingerprint(mol):
    if mol is None or GetMorganGenerator is None:
        return None
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    return generator.GetFingerprint(mol)


def tanimoto_similarity(mol_a, mol_b) -> float:
    if mol_a is None or mol_b is None or DataStructs is None:
        return 0.0
    fp_a = morgan_fingerprint(mol_a)
    fp_b = morgan_fingerprint(mol_b)
    if fp_a is None or fp_b is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def murcko_scaffold_smiles(mol) -> str:
    if mol is None or MurckoScaffold is None or Chem is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return ""
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold, canonical=True)


def murcko_scaffold_overlap(reactant_mol, product_mol) -> Dict[str, object]:
    if reactant_mol is None or product_mol is None:
        return {
            "reactant_scaffold": "",
            "product_scaffold": "",
            "murcko_shared_atoms": 0.0,
            "murcko_tanimoto": 0.0,
            "murcko_exact_match": False,
        }
    reactant_scaffold = murcko_scaffold_smiles(reactant_mol)
    product_scaffold = murcko_scaffold_smiles(product_mol)
    reactant_scaffold_mol = parse_smiles(reactant_scaffold)
    product_scaffold_mol = parse_smiles(product_scaffold)
    shared = safe_find_mcs([reactant_scaffold_mol, product_scaffold_mol])
    return {
        "reactant_scaffold": reactant_scaffold,
        "product_scaffold": product_scaffold,
        "murcko_shared_atoms": float(shared.get("shared_atoms") or 0.0),
        "murcko_tanimoto": round(tanimoto_similarity(reactant_scaffold_mol, product_scaffold_mol), 3),
        "murcko_exact_match": bool(reactant_scaffold and reactant_scaffold == product_scaffold),
    }


def safe_find_mcs(mols: Sequence[object]) -> Dict[str, float]:
    if not rdkit_is_available():
        return {"shared_atoms": 0.0, "shared_bonds": 0.0}
    valid_mols = [mol for mol in mols if mol is not None]
    if len(valid_mols) < 2:
        return {"shared_atoms": 0.0, "shared_bonds": 0.0}
    try:
        result = rdFMCS.FindMCS(
            valid_mols,
            timeout=3,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
        )
    except Exception:
        return {"shared_atoms": 0.0, "shared_bonds": 0.0}
    if getattr(result, "canceled", False):
        return {"shared_atoms": 0.0, "shared_bonds": 0.0}
    return {
        "shared_atoms": float(getattr(result, "numAtoms", 0) or 0.0),
        "shared_bonds": float(getattr(result, "numBonds", 0) or 0.0),
    }


def summarize_molecule(label: str, mol) -> str:
    if mol is None:
        return "{0}=invalid".format(label)
    return "{0}=formula:{1}, heavy_atoms:{2}, rings:{3}, hetero_atoms:{4}, exact_mw:{5:.2f}, tpsa:{6:.2f}, logp:{7:.2f}".format(
        label,
        mol_formula(mol),
        heavy_atom_count(mol),
        ring_count(mol),
        hetero_atom_count(mol),
        exact_mol_wt(mol),
        tpsa(mol),
        logp(mol),
    )


def build_reaction_profile(row: Dict[str, str]) -> Dict[str, object]:
    reactant_smiles = [value.strip() for value in (row.get("reactant1_smiles") or "", row.get("reactant2_smiles") or "") if value.strip()]
    product_smiles = (row.get("product_smiles") or "").strip()
    profile: Dict[str, object] = {
        "toolkit": "text",
        "rdkit_available": rdkit_is_available(),
        "reactant_smiles": reactant_smiles,
        "product_smiles": product_smiles,
        "invalid_smiles": [],
        "reactant_new_elements": [],
        "product_new_elements": [],
        "shared_atoms": 0.0,
        "shared_bonds": 0.0,
        "product_atom_coverage": 0.0,
        "reactant_atom_coverage": 0.0,
        "reactant_summary": "",
        "product_summary": "",
        "reactant_descriptors": {},
        "product_descriptors": {},
        "descriptor_delta": {},
        "fingerprint_tanimoto": 0.0,
        "murcko_tanimoto": 0.0,
        "murcko_shared_atoms": 0.0,
        "murcko_exact_match": False,
        "reactant_scaffold": "",
        "product_scaffold": "",
        "reactant_contributions": [],
    }
    if not rdkit_is_available():
        return profile

    reactant_mols = []
    for smiles in reactant_smiles:
        mol = parse_smiles(smiles)
        if mol is None:
            profile["invalid_smiles"].append(smiles)
            continue
        reactant_mols.append(mol)

    product_mol = parse_smiles(product_smiles)
    if product_smiles and product_mol is None:
        profile["invalid_smiles"].append(product_smiles)

    combined_reactants = combine_mols(reactant_smiles)
    profile["toolkit"] = "rdkit"
    profile["reactant_mol_count"] = len(reactant_mols)
    profile["product_valid"] = product_mol is not None
    profile["reactants_valid"] = len(reactant_mols) == len(reactant_smiles) and bool(reactant_smiles)
    profile["reactant_summary"] = summarize_molecule("reactants", combined_reactants)
    profile["product_summary"] = summarize_molecule("product", product_mol)

    if combined_reactants is not None and product_mol is not None:
        mcs = safe_find_mcs([combined_reactants, product_mol])
        profile.update(mcs)
        product_atoms = max(heavy_atom_count(product_mol), 1)
        reactant_atoms = max(heavy_atom_count(combined_reactants), 1)
        profile["product_atom_coverage"] = round(float(mcs["shared_atoms"]) / product_atoms, 3)
        profile["reactant_atom_coverage"] = round(float(mcs["shared_atoms"]) / reactant_atoms, 3)
        reactant_elements = set(element_set(combined_reactants))
        product_elements = set(element_set(product_mol))
        profile["product_new_elements"] = sorted(product_elements - reactant_elements)
        profile["reactant_new_elements"] = sorted(reactant_elements - product_elements)
        profile["combined_reactant_formula"] = mol_formula(combined_reactants)
        profile["product_formula"] = mol_formula(product_mol)
        profile["canonical_product_smiles"] = canonicalize_smiles(product_smiles)
        profile["canonical_reactant_smiles"] = ".".join(canonicalize_smiles(smiles) for smiles in reactant_smiles)
        profile["product_is_substructure_of_reactants"] = bool(combined_reactants.HasSubstructMatch(product_mol))
        profile["reactant_descriptors"] = descriptor_snapshot(combined_reactants)
        profile["product_descriptors"] = descriptor_snapshot(product_mol)
        profile["descriptor_delta"] = descriptor_delta(profile["reactant_descriptors"], profile["product_descriptors"])
        profile["fingerprint_tanimoto"] = round(tanimoto_similarity(combined_reactants, product_mol), 3)
        profile.update(murcko_scaffold_overlap(combined_reactants, product_mol))
        reactant_contributions = []
        for index, reactant_smiles_value in enumerate(reactant_smiles, 1):
            reactant_mol = parse_smiles(reactant_smiles_value)
            if reactant_mol is None:
                continue
            reactant_mcs = safe_find_mcs([reactant_mol, product_mol])
            reactant_heavy_atoms = max(heavy_atom_count(reactant_mol), 1)
            reactant_contributions.append(
                {
                    "reactant_index": index,
                    "reactant_smiles": reactant_smiles_value,
                    "shared_atoms": float(reactant_mcs.get("shared_atoms") or 0.0),
                    "shared_bonds": float(reactant_mcs.get("shared_bonds") or 0.0),
                    "product_atom_coverage": round(float(reactant_mcs.get("shared_atoms") or 0.0) / product_atoms, 3),
                    "reactant_atom_coverage": round(float(reactant_mcs.get("shared_atoms") or 0.0) / reactant_heavy_atoms, 3),
                    "fingerprint_tanimoto": round(tanimoto_similarity(reactant_mol, product_mol), 3),
                    "murcko_scaffold": murcko_scaffold_smiles(reactant_mol),
                }
            )
        profile["reactant_contributions"] = reactant_contributions
        profile["motifs"] = {
            "aryl_halide": has_substructure(combined_reactants, "[cX3][Cl,Br,I]"),
            "product_aryl_halide": has_substructure(product_mol, "[cX3][Cl,Br,I]"),
            "boronic_acid_or_ester": has_substructure(combined_reactants, "[B]([O])[O]") or has_substructure(combined_reactants, "[B]1OCC(C)(C)CO1"),
            "product_boronate": has_substructure(product_mol, "[B]([O])[O]") or has_substructure(product_mol, "[B]1OCC(C)(C)CO1"),
            "alkene": has_substructure(combined_reactants, "[CX3]=[CX3]") or has_substructure(combined_reactants, "[CX3]=[CX2]"),
            "benzyl_ether": has_substructure(combined_reactants, "[OX2][CH2][c]1[cH][cH][cH][cH][cH]1"),
            "product_benzyl_ether": has_substructure(product_mol, "[OX2][CH2][c]1[cH][cH][cH][cH][cH]1"),
            "acyl_halide": has_substructure(combined_reactants, "[CX3](=[OX1])[Cl,Br,F]"),
            "carboxyl": has_substructure(combined_reactants, "[CX3](=[OX1])[OX2H1,OX1-]"),
            "disulfide": has_substructure(combined_reactants, "[#16]-[#16]"),
            "phosphorus": has_substructure(combined_reactants, "[P]"),
            "silicon": has_substructure(combined_reactants, "[Si]"),
            "benzylic_halide": has_substructure(combined_reactants, "[CH2][Cl,Br,I]"),
            "product_aldehyde": has_substructure(product_mol, "[CH](=O)[c,C]") or has_substructure(product_mol, "[c,C]C=O"),
            "product_ketone": has_substructure(product_mol, "[#6][CX3](=O)[#6]"),
        }
        profile["summary"] = " | ".join(
            [
                profile["reactant_summary"],
                profile["product_summary"],
                "shared_atoms:{0}".format(int(profile["shared_atoms"])),
                "shared_bonds:{0}".format(int(profile["shared_bonds"])),
                "product_atom_coverage:{0:.2f}".format(profile["product_atom_coverage"]),
                "reactant_atom_coverage:{0:.2f}".format(profile["reactant_atom_coverage"]),
                "fingerprint_tanimoto:{0:.2f}".format(profile["fingerprint_tanimoto"]),
                "murcko_tanimoto:{0:.2f}".format(profile["murcko_tanimoto"]),
                "murcko_shared_atoms:{0}".format(int(profile["murcko_shared_atoms"])),
                "product_new_elements:{0}".format(",".join(profile["product_new_elements"]) or "none"),
            ]
        )
    else:
        profile["summary"] = " | ".join(
            [value for value in [profile["reactant_summary"], profile["product_summary"]] if value]
        )
    return profile
