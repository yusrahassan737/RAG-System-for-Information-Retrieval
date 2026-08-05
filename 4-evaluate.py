import torch
import pandas as pd
import importlib.util
import sys
from Eval_Questions import QUESTIONS

# Import functions from 3-retrieve.py
spec = importlib.util.spec_from_file_location("retrieve", "3-retrieve.py")
retrieve = importlib.util.module_from_spec(spec)
sys.modules["retrieve"] = retrieve
spec.loader.exec_module(retrieve)

compute_bm25 = retrieve.compute_bm25
dense_retrieval = retrieve.dense_retrieval

# Import pre_process() from 2-parse.py
spec2 = importlib.util.spec_from_file_location("parse", "2-parse.py")
parse = importlib.util.module_from_spec(spec2)
sys.modules["parse"] = parse
spec2.loader.exec_module(parse)

pre_process = parse.pre_process

embedding_data = torch.load("embeddings.pt", weights_only=False)

inverted_index = pd.read_csv("inverted_index.csv")
chunk_info = pd.read_csv("chunk_info.csv")
chunks = pd.read_csv("chunk_texts.csv")


def precision_at_10(results, relevant_chunks):
    """
    Compute Precision@10.

    results: DataFrame returned by BM25 or Dense Retrieval.
    relevant_chunks: List of correct chunk IDs.
    """

    top10 = results.head(10)["chunk"].tolist()

    relevant_found = 0

    for chunk in relevant_chunks:
        if chunk in top10:
            relevant_found += 1

    return relevant_found / 10




results = []

for question in QUESTIONS:

    bm25_scores = compute_bm25(
        pre_process(question["question"]),
        inverted_index,
        chunk_info
    )

    dense_scores = dense_retrieval(
        question["question"],
        chunks,
        embedding_data
    )

    bm25_p10 = precision_at_10(
        bm25_scores,
        question["relevant"]
    )

    dense_p10 = precision_at_10(
        dense_scores,
        question["relevant"]
    )

    results.append({
        "Question": question["id"],
        "BM25_P@10": bm25_p10,
        "Dense_P@10": dense_p10
    })

evaluation = pd.DataFrame(results)

print(evaluation)

evaluation.to_csv("evaluation.csv", index=False)

print("\nAverage BM25 Precision@10:", evaluation["BM25_P@10"].mean(),3)
print("Average Dense Precision@10:", evaluation["Dense_P@10"].mean(),3)
print("\nSaved Results evaluation.csv")