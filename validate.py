import json
import os

ARTIFACT_DIR = "artifacts"

ALLOWED_ANSWER_LABELS = {
    "grounded_answer",
    "insufficient_context",
    "conflicting_context"
}

ALLOWED_RETRIEVAL_STATUSES = {
    "hit",
    "partial_hit",
    "miss"
}

REQUIRED_FILES = [
    "chunks.json",
    "retrieval.json",
    "answers.json",
    "eval.json"
]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def check_files():
    for file in REQUIRED_FILES:
        path = os.path.join(ARTIFACT_DIR, file)
        if not os.path.exists(path):
            raise Exception(f"Missing required artifact: {file}")


def validate_retrieval():
    retrieval = load_json(f"{ARTIFACT_DIR}/retrieval.json")

    if len(retrieval) == 0:
        raise Exception("No retrieval results found")

    for record in retrieval:
        if len(record["top_k"]) < 3:
            raise Exception(f"{record['query_id']} has fewer than 3 retrieved chunks")

        for item in record["top_k"]:
            if not isinstance(item["score"], (int, float)):
                raise Exception(f"Non-numeric score in {record['query_id']}")


def validate_answers():
    answers = load_json(f"{ARTIFACT_DIR}/answers.json")
    retrieval = load_json(f"{ARTIFACT_DIR}/retrieval.json")

    retrieval_map = {}

    for r in retrieval:
        retrieval_map[r["query_id"]] = {
            x["chunk_id"] for x in r["top_k"]
        }

    for answer in answers:
        label = answer["answer_label"]

        if label not in ALLOWED_ANSWER_LABELS:
            raise Exception(f"Invalid answer label: {label}")

        if label == "grounded_answer":
            if not answer["citations"]:
                raise Exception(f"{answer['query_id']} missing citations")

        for cid in answer["used_chunk_ids"]:
            if cid not in retrieval_map[answer["query_id"]]:
                raise Exception(
                    f"{answer['query_id']} cites non-retrieved chunk {cid}"
                )


def validate_eval():
    eval_data = load_json(f"{ARTIFACT_DIR}/eval.json")

    if "summary" not in eval_data:
        raise Exception("Missing aggregate evaluation summary")

    for item in eval_data["evaluations"]:
        status = item["retrieval_status"]

        if status not in ALLOWED_RETRIEVAL_STATUSES:
            raise Exception(f"Invalid retrieval status: {status}")


def main():
    check_files()
    validate_retrieval()
    validate_answers()
    validate_eval()

    print("Validation passed successfully.")


if __name__ == "__main__":
    main()
