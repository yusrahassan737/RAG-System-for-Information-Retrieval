# Date: 2026/07/27
# Class: CP423
# Description: Script to parse each document and combine into a corpus
# Note: R09W0259.html, R96Q0050.html and R96S0106.html do not have an article element

from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import WordNetLemmatizer
import torch
from sentence_transformers import SentenceTransformer

# Constants
REM_WORDS = set(stopwords.words('english'))
REM_WORDS.update(set(string.punctuation))
REM_WORDS.update(["``", "''", "\n"])
LEMMATIZER = WordNetLemmatizer()
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text):
    # Split text
    chunks = []
    start = 0
    words = text.split()

    # Break up text, overlapping to capture context
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def pre_process(text):
    global REM_WORDS, LEMMATIZER
    # Clean, then tokenize.
    # Replace with space all characters except for alphabet, newlines or spaces before tokenizing
    text = re.sub(r'[^a-zA-Z\n]', ' ', text)
    text = text.lower()
    words = word_tokenize(text)

    # Remove stopwords and lemmatize
    filtered = [w for w in words if w.lower() not in REM_WORDS]
    lemmas = [LEMMATIZER.lemmatize(token) for token in filtered]

    return lemmas

# Add index info for one doc
def create_doc_index(doc_num, tokens):
    # Save term and document stats
    term_freqs = pd.Series(tokens).explode().value_counts()
    term_freqs = term_freqs.reset_index(name = "term_freq")
    term_freqs["doc_no"] = doc_num # assign the same doc_id to every token in this iteration of the loop
    return term_freqs

# Invert the complete index file
def construct_inverted_index(ind):
    # Sort postings
    ind = ind.sort_values(by = ["term", 'doc_no'])
    ind["doc_term_freqs"] = ind["doc_no"].astype(str) + " (" + ind["term_freq"].astype(str) + ")"

    # Group by term, aggregate the strings of the ids together per term
    inverted_ind = ind.groupby(by = 'term', as_index=False).agg({'doc_term_freqs': ', '.join})
    inverted_ind["doc_freq"] = inverted_ind["doc_term_freqs"].str.count(",") + 1
    inverted_ind["doc_freq"] = np.where(inverted_ind["doc_term_freqs"] != "", inverted_ind["doc_freq"], 0)

    return inverted_ind

def main():
    # Variables and set-up
    metadata = []
    chunk_lengths = []
    index = pd.DataFrame()
    chunk_texts = {"id": [], "text": []}

    all_reports = os.listdir("reports")
    doc_info = pd.read_csv("doc_info.csv")

    # Main loop
    for i in range(len(all_reports)):
        meta_temp = {"chunk_id": None, "doc_id": None, "title" : None, "creator" : None, "modified" : None, "occurence_date" : None, "release_date" : None}
        # Read document
        with open("reports/" + all_reports[i], "r", encoding = "utf-8") as f:
            text = f.read()
        soup = BeautifulSoup(text, "html.parser")

        # Save metadata information
        for j in ['title', 'creator', 'modified']:
            meta = soup.find('meta', attrs={'name': f'dcterms.{j}'})
            if meta:
                meta_temp[j] = meta.get('content')

        # Extract key times from report
        times = soup.find_all("time")
        if times:
            meta_temp["occurence_date"] = times[0].text
            if len(times) > 1:
                meta_temp["release_date"] = times[1].text

        # Main file content
        if soup.find('title'):
            text = soup.find('title').text + " "
        else:
            text = ""
        if soup.find('article'):
            text = soup.find('article').get_text(separator=" ")
        else:
            print(f"{all_reports[i]} does not have an article element")

        chunked = chunk_text(re.sub(r"\s+", " ", text).strip()) # remove all extra spaces
        for chunk_num, chunk in enumerate(chunked):
            # Save text/tokens for retrieval
            chunk_id = f"{i}_{chunk_num}"
            chunk_texts["id"].append(chunk_id)
            chunk_texts["text"].append(chunk) # original chunk text for dense retrieval    
            tokens = pre_process(chunk) # preprocessed chunk for BM25

            # Add to index and metadata
            chunk_lengths.append(len(tokens))
            term_frequencies = create_doc_index(chunk_id, tokens)
            index = pd.concat([index, term_frequencies], ignore_index=True)
            metadata.append({
                "chunk_id": chunk_id,
                "chunk_length": len(tokens),
                "doc_id": i,
                "title": meta_temp["title"],
                "creator": meta_temp["creator"],
                "modified": meta_temp["modified"],
                "occurence_date": meta_temp["occurence_date"],
                "release_date": meta_temp["release_date"]
            })

        # Testing if pre-processing is sufficient
        # with open("test.txt", "w") as f:
        #     f.write(text)
        #     f.write("\n" + str(tokens))

    # Get doc embeddings for dense retrieval
    chunk_embeddings = MODEL.encode(chunk_texts["text"], convert_to_tensor=True, show_progress_bar=True)

    # Create inverted index
    index = index.rename(columns={"index":"term"})
    index.to_csv("index.csv")
    inverted_index = construct_inverted_index(index)
    print(f"There are {len(inverted_index)} terms in the vocabulary/inverted index")
    pd.DataFrame(inverted_index).to_csv("inverted_index.csv", index = False)

    # Add metadata and document lengths to file
    chunk_info = pd.DataFrame(metadata)
    chunk_info = doc_info.merge(chunk_info, on = "doc_id")

    # Save everything
    chunk_info.to_csv("chunk_info.csv", index = False)
    chunk_texts = pd.DataFrame(chunk_texts)
    chunk_texts.to_csv("chunk_texts.csv", index = False)
    torch.save(chunk_embeddings, "embeddings.pt")

if __name__ == "__main__":
    main()