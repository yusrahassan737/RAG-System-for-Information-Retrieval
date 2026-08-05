# Date: 2026/08/04
# Class: CP423
# Description: Test the RAG model, by retrieving the ten most relevant chunks according to BM25 and Dense Retrieval

import torch
import pandas as pd
import math
import torch
import re
from sentence_transformers import util
import importlib.util
import sys
from sentence_transformers import SentenceTransformer
import ollama

# Constants
TOP_K = 10

# Compute BM25 for a given query for every document in matching postings lists
def compute_bm25(query, inverted_index, dl_lookup): # query is a list of tokens
    # Hyperparameters
    k1 = 1.2
    b = 0.75
    k3 = 1000

    # Variables
    N = len(dl_lookup["doc_id"]) # total docs in the collection
    AVG_DL = sum(dl_lookup["chunk_length"]) / N # average document length
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
            matching_docs = line["doc_term_freqs"].item().split(",")

            # Score one document at a time
            for j in matching_docs:
                id, tf = re.search(r"([^ ]+) \((\d+)\)", j).groups() # extract the doc id and tf
                id = id.strip()
                tf = int(tf)
                dl = dl_lookup.loc[dl_lookup["chunk_id"] == id, "chunk_length"].item() # get doc length of matching doc_id

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
    df = df.head(TOP_K)
    return df

def dense_retrieval(query, chunks, doc_em):
    # For dense retrieval, we use the full text instead of the pre-processed, tokenized verison,
    # to better capture context and hidden meanings
    MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_em = MODEL.encode(query, convert_to_tensor=True)
    sims = util.cos_sim(query_em, doc_em)[0] # compare embeddings

    # Get the ranked results
    top_results = torch.topk(sims, k=min(TOP_K, len(chunks)))
    results = pd.DataFrame({
        "document": top_results.indices.cpu().numpy(),
        "score": top_results.values.cpu().numpy()
    })
    results["rank"] = range(1, len(results) + 1)

    return results

def llm_response(query, top_chunks, context_chunks):
    joined_context = "\n".join([f"- {chunk}" for chunk in context_chunks])

    system_instruction = f"""You are a RAG-based question answering assistant for information
    regarding the Transportation Safety Board of Canada's Rail transportation safety investigations
    and reports. Use ONLY the provided context. If the answer is not contained in the context,
    reply exactly: "I don't know". When using information from a chunk, cite it using [chunk_id]."""

    user_prompt = f"Question: {query}\n\nContext Chunks:\n{joined_context}"

    response = ollama.chat(model='llama3.2',
        messages = [{'role': 'system', 'content': system_instruction},
            {'role': 'user', 'content': user_prompt}],
        options = {'temperature': 0.6}  # set lower than the default 0.8 to encourage more factual, deterministic outputs
    )

    return response['message']['content']

def main():
    # need to do long way of importing because of file naming
    spec = importlib.util.spec_from_file_location("pre_process", "2-parse_and_retrieve.py")
    my_module = importlib.util.module_from_spec(spec)
    sys.modules["pre_process"] = my_module
    spec.loader.exec_module(my_module)
    pre_process = my_module.pre_process
    inp_query = input()

    # Load necessary data
    chunk_embeddings = torch.load("embeddings.pt")
    inverted_index = pd.read_csv("inverted_index.csv")
    chunk_info = pd.read_csv("chunk_info.csv")
    chunks = pd.read_csv("chunk_texts.csv")

    # BM25
    bm25_scores = compute_bm25(pre_process(inp_query), inverted_index, chunk_info)
    bm25_scores.to_csv("bm25_results.csv", index= False)

    # Dense Retrieval
    dense_scores = dense_retrieval(inp_query, chunks, chunk_embeddings)
    dense_scores.to_csv("dense_results.csv", index= False)

    # LLM Results
    print(llm_response(inp_query, bm25_scores, chunks)) # bm25
    print(llm_response(inp_query, dense_scores, chunks)) # dense retrieval

if __name__ == "__main__":
    main()