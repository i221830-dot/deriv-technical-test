import os
import json
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# CONTROLLED VOCABULARIES
# ---------------------------
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

PIPELINE_STAGES = [
    "INIT",
    "DOCUMENTS_LOADED",
    "DOCUMENTS_CHUNKED",
    "INDEX_BUILT",
    "RETRIEVAL_COMPLETE",
    "ANSWERS_GENERATED",
    "EVALUATION_COMPLETE",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED"
]

state = "INIT"

KB_DIR = "kb"
ARTIFACT_DIR = "artifacts"
QUERIES_FILE = "queries.json"

Path(ARTIFACT_DIR).mkdir(exist_ok=True)


# ---------------------------
# DOCUMENT LOADING
# ---------------------------
def load_documents():
    global state
    docs = []

    for file in os.listdir(KB_DIR):
        if file.endswith(".txt"):
            path = os.path.join(KB_DIR, file)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            title_match = re.search(r"Title:\s*(.*)", content)
            section_match = re.search(r"Section:\s*(.*)", content)

            title = title_match.group(1).strip() if title_match else "Unknown"
            section = section_match.group(1).strip() if section_match else "Unknown"

            body = re.sub(r"Title:.*\n", "", content)
            body = re.sub(r"Section:.*\n", "", body).strip()

            docs.append({
                "title": title,
                "section": section,
                "body": body
            })

    state = "DOCUMENTS_LOADED"
    return docs


# ---------------------------
# CHUNKING
# ---------------------------
def chunk_documents(documents, chunk_size=120):
    global state
    chunks = []
    chunk_counter = 1

    for doc in documents:
        body = doc["body"]

        for start in range(0, len(body), chunk_size):
            end = min(start + chunk_size, len(body))
            text = body[start:end]

            chunks.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "doc_title": doc["title"],
                "section": doc["section"],
                "text": text,
                "start_char": start,
                "end_char": end
            })

            chunk_counter += 1

    with open(f"{ARTIFACT_DIR}/chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    state = "DOCUMENTS_CHUNKED"
    return chunks


# ---------------------------
# INDEX
# ---------------------------
def build_index(chunks):
    global state
    texts = [c["text"] for c in chunks]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)

    state = "INDEX_BUILT"
    return vectorizer, matrix


# ---------------------------
# RETRIEVAL
# ---------------------------
def retrieve(queries, chunks, vectorizer, matrix, top_k=3):
    global state
    retrieval_results = []

    for query in queries:
        q_vec = vectorizer.transform([query["question"]])
        scores = cosine_similarity(q_vec, matrix)[0]

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        top_chunks = []

        for rank, (idx, score) in enumerate(ranked, start=1):
            c = chunks[idx]

            top_chunks.append({
                "rank": rank,
                "chunk_id": c["chunk_id"],
                "doc_title": c["doc_title"],
                "score": float(score),
                "chunk_text": c["text"]
            })

        retrieval_results.append({
            "query_id": query["query_id"],
            "question": query["question"],
            "top_k": top_chunks
        })

    with open(f"{ARTIFACT_DIR}/retrieval.json", "w") as f:
        json.dump(retrieval_results, f, indent=2)

    state = "RETRIEVAL_COMPLETE"
    return retrieval_results


# ---------------------------
# ANSWERS
# ---------------------------
def generate_answers(retrieval_results):
    global state
    answers = []

    for result in retrieval_results:
        top = result["top_k"][0]

        if top["score"] < 0.05:
            answer = {
                "query_id": result["query_id"],
                "answer_label": "insufficient_context",
                "answer": "Insufficient context to answer.",
                "citations": [],
                "used_chunk_ids": []
            }
        else:
            citation = f'[{top["doc_title"]} §{top["chunk_id"]}]'

            answer = {
                "query_id": result["query_id"],
                "answer_label": "grounded_answer",
                "answer": f'{top["chunk_text"]} {citation}',
                "citations": [citation],
                "used_chunk_ids": [top["chunk_id"]]
            }

        answers.append(answer)

    with open(f"{ARTIFACT_DIR}/answers.json", "w") as f:
        json.dump(answers, f, indent=2)

    state = "ANSWERS_GENERATED"
    return answers


# ---------------------------
# EVALUATION
# ---------------------------
def evaluate(queries, retrieval_results):
    global state
    evaluations = []
    hits = partial = misses = 0

    for query, retrieval in zip(queries, retrieval_results):
        expected = query["expected_doc_titles"]
        retrieved_titles = [x["doc_title"] for x in retrieval["top_k"]]

        matched = any(title in retrieved_titles for title in expected)

        if matched:
            status = "hit"
            explanation = "Expected title found in top 3"
            hits += 1
        else:
            status = "miss"
            explanation = "Expected title not found"
            misses += 1

        evaluations.append({
            "query_id": query["query_id"],
            "expected_doc_titles": expected,
            "retrieved_doc_titles_top3": retrieved_titles,
            "retrieval_status": status,
            "matched_expected_title": matched,
            "explanation": explanation
        })

    summary = {
        "top3_hit_rate": hits / len(queries),
        "total_queries": len(queries),
        "hits": hits,
        "partial_hits": partial,
        "misses": misses
    }

    output = {
        "evaluations": evaluations,
        "summary": summary
    }

    with open(f"{ARTIFACT_DIR}/eval.json", "w") as f:
        json.dump(output, f, indent=2)

    state = "EVALUATION_COMPLETE"


# ---------------------------
# MAIN
# ---------------------------
def main():
    with open(QUERIES_FILE, "r") as f:
        queries = json.load(f)

    docs = load_documents()
    chunks = chunk_documents(docs)
    vectorizer, matrix = build_index(chunks)
    retrieval = retrieve(queries, chunks, vectorizer, matrix)
    generate_answers(retrieval)
    evaluate(queries, retrieval)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
