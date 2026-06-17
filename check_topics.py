import json
ok=True
n=0
for p in (json.loads(l) for l in open("runs/topic_tag_check/science_passages.jsonl",encoding="utf-8") if l.strip()):
    valid=set(p["topic_ids"])
    for q in p["questions"]:
        n+=1
        tid=q.get("topic_id")
        good = tid in valid
        ok = ok and good
        print(("OK " if good else "BAD"), q["question_id"], "topic_id=", tid, "topic=", repr(q.get("topic")), "passage_topics=", p["topic_ids"])
print(f"\n{n} questions checked")
print("ALL GOOD" if ok else "MISMATCH")
