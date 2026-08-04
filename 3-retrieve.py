# Date: August 1, 2026
# Class: CP423
# Description: Script to perform BM25 and Dense Retrieval
# Dense Retrieval: Use a pretrained embedding model (e.g., Word2Vec or Sentence Transformers), to generate document
# and query embeddings and retrieve documents based on embedding similarity.

import pandas as pd
import re
import math
import torch
from sentence_transformers import SentenceTransformer, util

# Compute BM25 for a given query for every document in matching postings lists
def compute_bm25(query, inverted_index, dl_lookup): # query is a list of tokens
    # Hyperparameters
    k1 = 1.2
    b = 0.75
    k3 = 1000

    # Variables
    N = len(dl_lookup["doc_id"]) # total docs in the collection
    AVG_DL = sum(dl_lookup["doc_length"]) / N # average document length
    bm25_scores = {}

    # Look through one token of the query at a time
    for i in query:
        # Get query term frequencies
        qtf = query.count(i)

        # Get the postings list for each token in the query
        line = inverted_index[inverted_index["term"] == i]

        # Only compute score if the term exists in the collection
        if len(line) != 0:
            df = line["doc_freq"].item()
            idf = math.log((N - df + 0.5) / (df + 0.5))
            matching_docs = line["doc_info"].item().split(",")

            # Score one document at a time
            for j in matching_docs:
                id, tf = re.search(r"(\d+) \((\d+)\)", j).groups() # extract the doc id and tf
                id = int(id.strip())
                tf = int(tf)
                dl = dl_lookup.iloc[id, -1] # get doc length of matching doc_id

                formula = idf * (((tf * (k1 + 1))
                / (tf + k1 * (1 - b + b * (dl / AVG_DL))))
                * (((k3 + 1) * qtf) / (k3 + qtf))
                )

                # Add score to previous if it exists
                if id not in bm25_scores:
                    bm25_scores[id] = formula
                else:
                    bm25_scores[id] += formula
    df = pd.DataFrame(bm25_scores.items(), columns=["document", "score"])
    df = df.sort_values("score", ascending=False)
    df["score"] = df["score"].round(3)
    df["rank"] = list(range(1, (len(df) + 1)))
    df = df.head(100)

    return df

inverted_index = pd.read_csv("inverted_index.csv")
doc_info = pd.read_csv("doc_info_updated.csv")
scores_df = compute_bm25("ottawa", inverted_index, doc_info)
scores_df.to_csv("results.csv")

def dense_retrieval():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
