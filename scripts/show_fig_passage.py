import json, sys
path = sys.argv[1]
d = json.loads(open(path, encoding="utf-8").read().splitlines()[0])
print("SECTION:", d["section"], "| TOPICS:", d.get("topic_ids"))
print("\nPASSAGE:\n", d["passage_text"])
print("\nTABLE:", d.get("table_markdown"))
print("\nPASSAGE FIGURES:")
for f in d.get("figures", []):
    print(" ", f["figure_type"], "|", f.get("figure_id"), "|", json.dumps(f.get("smiles") or f.get("plot")))
print("\n=== QUESTIONS ===")
for q in d["questions"]:
    print("\n[" + q["answer_basis"] + " | " + q["skill_tested"] + " | " + q["difficulty"] + "]")
    print(q["stem"])
    for f in q.get("figures", []):
        print("  FIGURE:", f["figure_type"], "|", json.dumps(f.get("smiles") or f.get("plot")))
    print("  Ans:", q["correct_answer"])
