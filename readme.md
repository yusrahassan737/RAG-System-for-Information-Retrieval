## CP423 Course Project
# RAG System for Canada Rail Transportation Safety Reports

## Overview
Developed for the course CP423-B at Wilfrid Laurier University

The project objective is to build a Retrieval Augmented Generation System which can answer questions by retrieving  from the official TSB safety reports. Instead of just reliance on the LLM’s internal storage the system, works to retrieve the relevant reports before the response is generated. 

The project compares two retrieval approaches:

- BM25 (lexical retrieval)
- Dense Retrieval (semantic retrieval)

Both retrieval methods will ultimately be evaluated using the same LLM.

## Dataset

Corpus:
- Transportation Safety Board (TSB) Rail Investigation Reports

Official Website:
https://www.tsb.gc.ca/eng/rapports-reports/rail/index.html

Approximately 482 publicly available English investigation reports are used in this project.

Each report contains:

- Investigation Number
- Title
- Report Content
- Metadata
- Publication Information

##Features Include

BeautifulSoup Parsing integration
Dense Semantic Retrieval with Pretrained Sentence Transformers
Retrieval Augmented Generation using Llama 3.2 (Ollama)
BM25 lexical retrieval 
Metadata extraction from TSB investigation reports
Text Preprocessing using:
	- Tokenization
              	- Stop Word removal 
- Lemmatization
Automatic chunking with windows overlapping



# Repository Structure 


├── reports/                 # Downloaded TSB HTML investigation reports

├── 1-crawl.py               # Downloads investigation reports
├── 2-parse.py               # Parses reports, preprocesses text, builds corpus
├── 3-retrieve.py            # BM25, Dense Retrieval, and LLM generation
├── 4-evaluate.py            # Precision@10 evaluation


├── Eval_Questions.py        # Gold-standard evaluation questions

├── chunk_info.csv           # Chunk metadata
├── chunk_texts.csv          # Original chunk text
├── inverted_index.csv       # BM25 inverted index
├── index.csv                # Intermediate term index
├── embeddings.pt            # Dense document embeddings

├── bm25_results.csv         # BM25 retrieval output
├── dense_results.csv        # Dense retrieval output
├── evaluation.csv           # Precision@10 evaluation results

├── requirements.txt
└── README.md

---

# Libraries 

- BeautifulSoup
- Pandas
- NumPy
- NLTK
- PyTorch
- Sentence Transformers
- Ollama
- Llama 3.2


# Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download  required NLTK resources:

```bash
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

# Project

### Crawl the Reports

```bash
python3 1-crawl.py
```

Downloads allreports.


### Corpus Parsing and Building 

```bash
python3 2-parse.py
```


### Retrieve Documents

```bash
python3 3-retrieve.py
```

Enter a question to ask and you will be returned with 

- Retrieval of the Top-10 BM25 chunks
- Retrieve of the Top-10 Dense Retrieval chunks
- Generate responses using the LLM


### Evaluate 

```bash
python3 4-evaluate.py
```

Computes Precision@10 for both BM25 and Dense Retrieval 


Evaluation results are saved to:
evaluation.csv


