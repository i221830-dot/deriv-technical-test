# Mini RAG Pipeline

## Overview
This project implements a deterministic mini Retrieval-Augmented Generation (RAG) pipeline.

Stages:
- Document ingestion
- Deterministic chunking
- TF-IDF retrieval
- Citation-grounded answer generation
- Retrieval evaluation
- Validation checks
- Grounding verification

## Tech Stack
- Python
- scikit-learn
- Flask

## Run
Install dependencies:

pip install -r requirements.txt

Run pipeline:

python pipeline.py

Validate:

python validate.py

Grounding check:

python grounding_check.py

## Design Choices
- TF-IDF used for deterministic retrieval
- No LLM dependency for reproducibility
- Citation-strict grounded answers
- JSON artifact generation for evaluator inspection
