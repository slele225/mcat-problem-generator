import json
d = json.loads(open("runs/sp_test/science_passages.jsonl").read().splitlines()[0])
print("SECTION:", d["section"])
print("TOPICS:", d.get("topic_ids"))
print("\nPASSAGE:\n", d["passage_text"])
print("\nTABLE:", d.get("table_markdown"))
print("\n===", len(d["questions"]), "QUESTIONS ===\n")
for q in d["questions"]:
    print("[" + q["answer_basis"] + " | " + q["skill_tested"] + " | " + q["difficulty"] + "]")
    print(q["stem"])
    print("Ans:", q["correct_answer"], "\n")
