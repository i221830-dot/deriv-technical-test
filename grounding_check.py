import json

with open("artifacts/answers.json") as f:
    answers = json.load(f)

with open("artifacts/retrieval.json") as f:
    retrieval = json.load(f)

retrieval_map = {}

for r in retrieval:
    retrieval_map[r["query_id"]] = {
        item["chunk_id"]: item["chunk_text"]
        for item in r["top_k"]
    }

results = []

for ans in answers:
    qid = ans["query_id"]
    citations_valid = True
    support_found = True

    for cid in ans["used_chunk_ids"]:
        if cid not in retrieval_map[qid]:
            citations_valid = False
            support_found = False

    results.append({
        "query_id": qid,
        "citations_valid": citations_valid,
        "support_found": support_found,
        "status": "pass" if citations_valid and support_found else "fail"
    })

with open("artifacts/grounding_check.json", "w") as f:
    json.dump(results, f, indent=2)

print("Grounding check complete.")
