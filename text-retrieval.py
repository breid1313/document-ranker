import argparse
import os
import nltk
from algorithms.document_ranker import Ranker

parser = argparse.ArgumentParser()
parser.add_argument("--query", "--Q", help="Query to submit to the \"database\"", required=False)
parser.add_argument("--algorithm", "-A", help="Search algorithm to use. Options: bm25, jm, dirichlet, pln, default", required=False)
args = parser.parse_args()


data = nltk.corpus.webtext

# tokenize the query
tokenizer = nltk.tokenize.word_tokenize
query_tokens = tokenizer(args.query)
print("Tokenized query:")
print(query_tokens)

# read list of files
documents = data.fileids()

# sanity check, list file IDs
print("found files")
print(documents)

corpus = []
for document in documents:
    wordList = data.words(document)
    print(type(wordList))
    print("words in doc:")
    print(wordList)
    corpus.append(wordList)
    len(wordList)

engine = Ranker(corpus, alpha=0.25, b=0.75, k=1.2, mu=0.75)

result = engine.search(query_tokens, algorithm=args.algorithm)

print(result)