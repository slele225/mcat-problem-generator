"""Offline unit + real-data tests for the ionic/disconnected SMILES backstop.

No API calls (RDKit + matplotlib only). Run from the repo root:
    python tests/test_ionic_figures.py

Covers the Part A render-time backstop in src/figures.py:
  * _mol_is_drawable_structure — disconnected/ionic/metal species -> formula card;
    single connected covalent molecules (incl. zwitterions) -> structure.
  * _looks_like_formula / _format_formula / _molecule_formula_entry — the formula
    text shown on the card (prefer a clean label formula, else RDKit-derived).
  * end-to-end render of an ionic and a covalent spec (must both produce a PNG).
  * a scan of the shipped beta bank: EXACTLY the CuSO4.5H2O figure converts.
"""
import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Formula cards print Unicode subscripts (CuSO₄·5H₂O); make stdout UTF-8 so this
# test runs directly on a Windows console (cp1252) without an encode crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import figures as F  # noqa: E402
from src.schemas import FigureSpec  # noqa: E402

from rdkit import Chem  # noqa: E402


def _drawable(smiles: str) -> bool:
    return F._mol_is_drawable_structure(Chem.MolFromSmiles(smiles))


# (smiles, expected_drawable) — True = nice structure, False = formula card.
DRAWABLE_CASES = [
    ("OC(=O)c1ccccc1", True),                       # benzoic acid (covalent)
    ("Cc1ccccc1", True),                            # toluene
    ("Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C1O", True),  # ATP (connected, neutral)
    ("C(C(=O)[O-])[NH3+]", True),                   # glycine zwitterion (one fragment)
    ("CC(C)(C)Br", True),                           # tert-butyl bromide
    ("[Cu+2].[O-]S(=O)(=O)[O-].O.O.O.O.O", False),  # CuSO4.5H2O (7 frags + metal)
    ("[Na+].[Cl-]", False),                         # NaCl (2 frags + metal)
    ("[K+].[O-][Mn](=O)(=O)=O", False),             # KMnO4 (2 frags + metal)
    ("[Na+].[O-]C(=O)C", False),                    # sodium acetate (salt, metal)
]

# (label, smiles, expected_formula_display, expected_name)
FORMULA_ENTRY_CASES = [
    # Label already a clean formula -> use it (subscripted), no redundant name.
    ("CuSO4·5H2O", "[Cu+2].[O-]S(=O)(=O)[O-].O.O.O.O.O", "CuSO₄·5H₂O", None),
    # Label is a NAME -> derive from RDKit; keep the name as the sub-line.
    ("sodium chloride", "[Na+].[Cl-]", "NaCl", "sodium chloride"),
    ("copper sulfate pentahydrate", "[Cu+2].[O-]S(=O)(=O)[O-].O.O.O.O.O",
     "CuO₄S·5H₂O", "copper sulfate pentahydrate"),
]

# _looks_like_formula: (text, expected)
FORMULA_SNIFF_CASES = [
    ("CuSO4·5H2O", True),
    ("NaCl", True),
    ("Ca(OH)2", True),
    ("Al2(SO4)3", True),
    ("Copper(II) sulfate pentahydrate", False),  # has spaces -> name, not formula
    ("tert-butyl bromide", False),
    ("ATP", False),                               # 'A' is not an element symbol
    ("benzoic acid", False),
]


def run_unit():
    fails = []

    for smi, want in DRAWABLE_CASES:
        got = _drawable(smi)
        ok = got == want
        kind = "structure" if want else "formula card"
        print(f"  [{'PASS' if ok else 'FAIL'}] drawable={got!s:5} (want {kind:12}) | {smi}")
        if not ok:
            fails.append(("drawable mismatch", smi, got, want))

    for text, want in FORMULA_SNIFF_CASES:
        got = F._looks_like_formula(text)
        ok = got == want
        print(f"  [{'PASS' if ok else 'FAIL'}] looks_like_formula={got!s:5} (want {want!s:5}) | {text!r}")
        if not ok:
            fails.append(("formula-sniff mismatch", text, got, want))

    for label, smi, want_f, want_n in FORMULA_ENTRY_CASES:
        m = Chem.MolFromSmiles(smi)
        got_f, got_n = F._molecule_formula_entry(SimpleNamespace(label=label), m)
        ok = got_f == want_f and got_n == want_n
        print(f"  [{'PASS' if ok else 'FAIL'}] entry={got_f!r},{got_n!r} (want {want_f!r},{want_n!r}) | label={label!r}")
        if not ok:
            fails.append(("formula-entry mismatch", label, (got_f, got_n), (want_f, want_n)))

    return fails


def run_render():
    """End-to-end: an ionic and a covalent spec must each render a non-empty PNG."""
    fails = []
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        specs = {
            "ionic": FigureSpec(
                figure_id="t_ionic", figure_type="smiles",
                caption="Copper(II) sulfate pentahydrate",
                smiles={"molecules": [
                    {"smiles": "[Cu+2].[O-]S(=O)(=O)[O-].O.O.O.O.O", "label": "CuSO4·5H2O"}]},
            ),
            "covalent": FigureSpec(
                figure_id="t_covalent", figure_type="smiles", caption="Benzoic acid",
                smiles={"molecules": [{"smiles": "OC(=O)c1ccccc1", "label": "benzoic acid"}]},
            ),
        }
        for name, spec in specs.items():
            try:
                F.render_figure(spec, out)
                p = out / f"{spec.figure_id}.png"
                ok = p.exists() and p.stat().st_size > 0
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"  [FAIL] render {name}: raised {e}")
                fails.append(("render raised", name, str(e)))
                continue
            print(f"  [{'PASS' if ok else 'FAIL'}] render {name} -> {p.name} ({p.stat().st_size if p.exists() else 0} B)")
            if not ok:
                fails.append(("render produced no file", name))
    return fails


# In the shipped beta bank, ONLY the CuSO4.5H2O figure is ionic/disconnected.
EXPECTED_CARDS = {"SP_CP_4E_034_01_pf01"}


def run_realdata():
    bank = Path(__file__).resolve().parent.parent / "runs" / "beta_bank_v1" / "science_passages.jsonl"
    if not bank.exists():
        print("  (skip: beta_bank_v1/science_passages.jsonl not found)")
        return set(), set()
    cards, structures = set(), set()
    for line in open(bank, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        for fig in (rec.get("figures", []) or []):
            if fig.get("figure_type") != "smiles":
                continue
            mols = fig["smiles"]["molecules"]
            is_card = not all(_drawable(mol["smiles"]) for mol in mols)
            (cards if is_card else structures).add(fig["figure_id"])
    print(f"  bank smiles: {len(cards)} formula card(s), {len(structures)} structure(s)")
    for fid in sorted(cards):
        print(f"    formula card: {fid}")
    return cards, structures


if __name__ == "__main__":
    print("=== UNIT CASES ===")
    fails = run_unit()
    print("\n=== END-TO-END RENDER ===")
    fails += run_render()
    print("\n=== REAL DATA (runs/beta_bank_v1) ===")
    cards, structures = run_realdata()

    print("\n=== SUMMARY ===")
    missing = EXPECTED_CARDS - cards
    extra = cards - EXPECTED_CARDS
    if cards or structures:
        if missing:
            print(f"  REAL-DATA MISS (ionic not caught): {sorted(missing)}")
        if extra:
            print(f"  REAL-DATA EXTRA (covalent wrongly converted): {sorted(extra)}")
        if not missing and not extra:
            print(f"  real-data: exactly {sorted(EXPECTED_CARDS)} converted, all others kept as structures")
        if missing or extra:
            fails.append(("real-data card set mismatch", sorted(cards)))

    if fails:
        print(f"\n  {len(fails)} FAILURE(S):")
        for f in fails:
            print("   ", f)
        sys.exit(1)
    print("\nALL TESTS PASSED")
