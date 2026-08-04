# Date: 2026/07/27
# Class: CP423
# Description: Script to parse each document and combine into a corpus
# Note:
# R09W0259.html, R96Q0050.html and R96S0106.html do not have an article element

from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import WordNetLemmatizer

# Constants
REM_WORDS = set(stopwords.words('english'))
REM_WORDS.update(set(string.punctuation))
REM_WORDS.update(["``", "''", "\n"])
LEMMATIZER = WordNetLemmatizer()

# Variables and set-up
metadata = {"title" : [], "creator" : [], "modified" : [], "occurence_date" : [], "release_date" : []}
doc_lengths = []

all_reports = os.listdir("reports")
doc_info = pd.read_csv("doc_info.csv")

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
    ind["doc_info"] = ind["doc_no"].astype(str) + " (" + ind["term_freq"].astype(str) + ")"

    # Group by term, aggregate the strings of the ids together per term
    inverted_ind = ind.groupby(by = 'term', as_index=False).agg({'doc_info': ', '.join})
    inverted_ind["doc_freq"] = inverted_ind["doc_info"].str.count(",") + 1
    inverted_ind["doc_freq"] = np.where(inverted_ind["doc_info"] != "", inverted_ind["doc_freq"], 0)

    return inverted_ind

def main():
    index = pd.DataFrame()
    documents = []
    for i in range(len(all_reports)):
        # Read document
        with open("reports/" + all_reports[i], "r", encoding = "utf-8") as f:
            text = f.read()
        soup = BeautifulSoup(text, "html.parser")

        # Save metadata information
        for j in ['title', 'creator', 'modified']:
            meta = soup.find('meta', attrs={'name': f'dcterms.{j}'})
            if meta:
                metadata[j].append(meta.get('content'))
            else:
                metadata[j].append(None)

        # Extract key times from report
        times = soup.find_all("time")
        if times:
            metadata["occurence_date"].append(times[0].text)
            if len(times) > 1:
                metadata["release_date"].append(times[1].text)
            else:
                metadata["release_date"].append(None)
        else:
            metadata["occurence_date"].append(None)
            metadata["release_date"].append(None)

        # Main file content
        if soup.find('title'):
            text = soup.find('title').text + " "
        else:
            text = ""
        if soup.find('article'):
            text = soup.find('article').get_text(separator=" ")
        else:
            print(f"{all_reports[i]} does not have an article element")

        documents.append(re.sub(r"\s+", " ", text).strip()) # keep full text for dense retrieval

        # Pre-process the text, then get term-doc information
        tokens = pre_process(text)
        doc_lengths.append(len(tokens))
        term_frequencies = create_doc_index(i, tokens)

        # Testing if pre-processing is sufficient
        # with open("test.txt", "w") as f:
        #     f.write(text)
        #     f.write("\n" + str(tokens))

        # Add this document's terms to the overall index
        index = pd.concat([index, term_frequencies])

    # Create inverted index
    index = index.rename(columns={"index":"term"})
    index.to_csv("index.csv")
    inverted_index = construct_inverted_index(index)
    print(f"There are {len(inverted_index)} terms in the vocabulary/inverted index")
    pd.DataFrame(inverted_index).to_csv("inverted_index.csv", index = False)

    # Add metadata and document lengths to file
    for i in metadata: 
        doc_info[f"modified_{i}"] = metadata[i]
    doc_info["doc_length"] = doc_lengths

    # Save
    doc_info.to_csv("doc_info_updated.csv")

if __name__ == "__main__":
    main()