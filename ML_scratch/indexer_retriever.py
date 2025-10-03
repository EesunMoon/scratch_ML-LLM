"""
We have a list of documents. 
We want to build an index that maps keywords to documents containing them. 
Then, given a query keyword, we can efficiently retrieve all matching documents.

docs = [ "Python is great for data science", 
"C++ is a powerful language", 
"Python supports OOP and functional programming", 
"Weather today is sunny", 
"Weather forecast shows rain" ]
"""
# -- keyword-based retrieval --#
# 1. preprocess: tokenize, lowercase, remove stopwords, (optional) stemming/lemmatization
# 2. indexing: build an inverted index mapping keywords to document IDs
# 3. searching: given a query, preprocess it and look up in the index

#-- similarity-based retrieval --#
# 1. preprocess
# 2. embedding: convert documents and query into vectors (e.g., using TF-IDF, Word2Vec, BERT)
# 3. searching: compute similarity (e.g., cosine similarity) between query vector and document vectors, return top-k most similar documents
from collections import defaultdict
import numpy as np

class Retriever:
    def __init__(self, docs, stopwords):
        self.docs = docs
        self.stopwords = stopwords
        self.tokenizer = set()
        self.inverted_index = defaultdict(set) # keyword -> set(doc_ids)
        self.words_to_id = {} # word: id
        self.embeddings = []
        self.vocab_size = 0

    def tokenize(self, text):
        # 1) lowercase
        text = text.lower()
        # 2) split by space
        tokens = text.split()
        # 3) remove stopwords
        tokens = [t for t in tokens if t not in self.stopwords]
        tokens = list(set(tokens)) # unique tokens
        # 4) (optional) stemming/lemmatization
        tokens = [t[:-3] if t.endswith('ing') else (t[:-2] if t.endswith('ed') else t) for t in tokens]

        return tokens
    
    def build_tokenizer(self, docs):
        for doc in docs:
            tokens = self.tokenize(doc)
            self.vocab_size = max(self.vocab_size, len(tokens))
            self.tokenizer.update(tokens)

    def indexer(self, docs):
        for doc_id, doc in enumerate(docs):
            tokens = self.tokenize(doc)
            for token in tokens:
                self.inverted_index[token].add(doc_id)

    def search(self, query):
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return set()
        
        results = set()
        for token in query_tokens:
            if token in self.inverted_index:
                results.update(self.inverted_index[token])
        
        for doc_id in results:
            print(f"Doc ID: {doc_id}, Content: {self.docs[doc_id]}")

        return results
    
    def cosine_similarity(self, vec1, vec2):
        # dot product / norms
        eps = 1e-10
        return (vec1 @ vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2)+eps)
    
    def encoder(self, docs):
        self.build_tokenizer(docs)

        for idx, word in enumerate(sorted(list(self.tokenizer))):
            self.words_to_id[word] = idx
        self.words_to_id["<UNK>"] = len(self.words_to_id)
    
    def embedding(self, docs):
        # [docs_num, max(vocab_size)]
        # can use tf-idf, word2vec, pretrained bert

        self.encoder(docs)
        
        for doc in docs:
            vec = np.zeros(self.vocab_size)
            tokens = self.tokenize(doc)
            for i, token in enumerate(tokens):
                if token in self.words_to_id:
                    vec[i] = self.words_to_id[token]
                else:
                    vec[i] = self.words_to_id["<UNK>"]
            self.embeddings.append(vec)
        self.embeddings = np.array(self.embeddings)
    
    def retrieve_similarity(self, query, topK):
        query_vec = self.embedding(query)
        sim = [] # (doc_id, similarity)
        for doc_id, doc_vec in enumerate(self.embeddings):
            cos_sim = self.cosine_similarity(query_vec, doc_vec)
            sim.append(doc_id, cos_sim)

        sim.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sim[:topK]]

