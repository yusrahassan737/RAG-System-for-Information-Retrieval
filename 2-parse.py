# Date: 2026/07/27
# Class: CP423
# Description: Script to parse each document and combine into a corpus
# Note:
# R09W0259.html does not have an article element
# R96Q0050.html does not have an article element
# R96S0106.html does not have an article element

from bs4 import BeautifulSoup
import os
import pandas as pd

doc_info = pd.read_csv("doc_info.csv")
metadata = {"title" : [], "creator" : [], "modified" : []}
doc_lengths = []
all_reports = os.listdir("reports")
index = pd.DataFrame()

def pre_process(text):
    # Clean, then tokenize. Next, remove stopwords and lemmatize.
    return text

for report in all_reports:
    # Read document
    with open("reports/" + report, "r", encoding = "utf-8") as f:
        text = f.read()
    soup = BeautifulSoup(text, "html.parser")

    # Save metadata information
    for i in ['title', 'creator', 'modified']:
        meta = soup.find('meta', attrs={'name': f'dcterms.{i}'})
        if meta:
            metadata[i] = meta

    # Main file content
    if soup.find('title'):
        text = soup.find('title').text + " "
    else:
        text = ""
    if soup.find('article'):
        text = soup.find('article').text
    else:
        print(f"{report} does not have an article element")

    # Pre-process the text, then get term-doc information
    tokens = pre_process(text)
    doc_lengths.append(len(tokens))

# Add metadata and document lengths to file
for i in metadata: 
    doc_info[f"modified_{i}"] = metadata[i]
doc_info["doc_length"] = doc_lengths

# Save 
doc_info.to_csv("doc_info_updated.csv")