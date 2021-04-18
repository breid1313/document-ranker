import math
import nltk

class Ranker(object):
    def __init__(self, corpus, alpha=1, b=1, k=1, mu=1):
        self.corpus = corpus
        self.ALPHA = alpha
        self.B = b
        self.K = k
        self.MU = mu
        self.avdl = self.setAVDL()
    
    ########
    # setter methods
    ########
    def setAVDL(self, corpus=None):
        if not corpus:
            corpus = self.corpus
        total = 0
        for doc in corpus:
            total += float(len(doc))
        avdl = total / len(corpus)
        return avdl

    def setAlpha(self, alpha):
        self.ALPHA = alpha
        return
    
    def setB(self, b):
        self.B = b
        return

    def setK(self, k):
        self.K = k
        return

    def setMu(self, mu):
        self.MU = mu
        return

    ########
    # getter methods
    ########
    def getAVDL(self):
        return self.avdl

    def getAlpha(self):
        return self.ALPHA
    
    def getB(self):
        return self.B

    def getK(self):
        return self.K

    def getMu(self, mu):
        return self.MU

    ########
    # retreival models
    ########
    def dotProduct(self, query, document):
        """
        Take a query and document vectors (list) and computes the dot product
        :param document: tokenized document (list of words)
        """
        # this is quite expensive
        # calculate underlying vocabluary
        # assume this to be the unique words in the corpus
        vocab = [] 
        for doc in self.corpus:
            vocab += list(doc)
        unique = []
        for word in vocab:
            if word not in unique:
                unique.append(word)
        
        # build query and doc vectors based on vocab
        fd_query = nltk.FreqDist(query)
        fd_document = nltk.FreqDist(document)
        query_vector = []
        doc_vector = []
        for word in unique:
            query_vector.append(fd_query[word])
            doc_vector.append(fd_document[word])

        # calculate dot product
        dot = 0
        for i in range(len(query)):
            dot += query_vector[i] * doc_vector[i]
        return dot

    def bm25(self, query, document):
        """
        Calculate the BM25 score of a document
        :param document: tokenized document (list of words)
        """
        doc_len = len(document)
        fd_query = nltk.FreqDist(query)
        fd_document = nltk.FreqDist(document)

        # initialize score
        score = 0
        
        for word in query:
            if word in document:
                ##
                # get some word-specific values
                ##
                # calculate document frequency
                docuemnt_frequency = 0
                for doc in self.corpus:
                    if word in doc:
                        docuemnt_frequency += 1
                    else:
                        continue
                # get query count        
                query_count = fd_query[word]
                # get document count
                document_count = fd_document[word]
                ##
                # carfeully build each term
                ##
                numerator = (self.K + 1)*document_count
                demoninator = document_count + self.K * (1 - self.B + self.B * (doc_len/self.avdl))
                log_term = math.log(len(self.corpus) + 1 / docuemnt_frequency)
                ##
                # calculate the score for this word and add to the total
                ##
                score += query_count * (numerator / demoninator) * log_term
        return score


    def pivoted_length_normalization(self, query, document):
        """
        Calculate the pln score of a document
        :param document: tokenized document (list of words)
        """
        doc_len = len(document)
        fd_query = nltk.FreqDist(query)
        fd_document = nltk.FreqDist(document)

        # initialize score
        score = 0
        
        for word in query:
            if word in document:
                ##
                # get some word-specific values
                ##
                # calculate document frequency
                docuemnt_frequency = 0
                for doc in self.corpus:
                    if word in doc:
                        docuemnt_frequency += 1
                # get query count        
                query_count = fd_query[word]
                # get document count
                document_count = fd_document[word]
                ##
                # carfeully build each term
                ##
                numerator = math.log(1 + math.log(1 + document_count))
                demoninator = 1 - self.B + self.B * (doc_len / self.avdl)
                log_term = math.log(len(self.corpus) + 1 / docuemnt_frequency)
                score += query_count * (numerator / demoninator) * log_term
        return score

    def jm_smoothing(self, query, document):
        """
        Calculate the jm smoothing score of a document
        :param document: tokenized document (list of words)
        """
        doc_len = len(document)
        fd_query = nltk.FreqDist(query)
        fd_document = nltk.FreqDist(document)

        # initialize score
        score = 0
        
        for word in query:
            if word in document:
                ##
                # get some word-specific values
                ##
                # calculate document frequency
                docuemnt_frequency = 0
                for doc in self.corpus:
                    if word in doc:
                        docuemnt_frequency += 1
                # get query count        
                query_count = fd_query[word]
                # get document count
                document_count = fd_document[word]
                ##
                # carfeully build each term
                ##
                # Assumption: p(w|C) = 0.02 for all words.
                # TODO refine above
                log_term = math.log(1 + (1 - self.ALPHA)/self.ALPHA * document_count/(doc_len * 0.02))
                score += query_count * log_term + len(query) * math.log(self.ALPHA)
        return score


    def dirichlet_smoothing(self, query, document):
        """
        Calculate the dirichlet smoothing score of a document
        :param document: tokenized document (list of words)
        """
        doc_len = len(document)
        fd_query = nltk.FreqDist(query)
        fd_document = nltk.FreqDist(document)

        # initialize score
        score = 0
        
        for word in query:
            if word in document:
                ##
                # get some word-specific values
                ##
                # calculate document frequency
                docuemnt_frequency = 0
                for doc in self.corpus:
                    if word in doc:
                        docuemnt_frequency += 1
                # get query count        
                query_count = fd_query[word]
                # get document count
                document_count = fd_document[word]
                ##
                # carfeully build each term
                ##
                # Assumption: p(w|C) = 0.02 for all words.
                # TODO refine above
                log_term = math.log(1 + document_count/(self.MU * 0.02))
                score += query_count * log_term + len(query) * math.log(self.MU / (self.MU + doc_len))
        return score


    def search(self, query, algorithm="Default"):
        """
        search a corpus based on a query and return the top document
        """
        bestScore = -math.inf
        bestDoc = None
        rank = None
        method = None

        if algorithm.lower() == "bm25":
            rank = self.bm25
            method = "BM25/Okapi"
        elif algorithm.lower() == "jm":
            rank = self.jm_smoothing
            method = "JM Smoothing"
        elif algorithm.lower() == "dirichlet":
            rank = self.dirichlet_smoothing
            method = "Dirichlet Prior Smoothing"
        elif algorithm.lower() == "pln":
            rank = self.pivoted_length_normalization
            method = "Pivoted Length Normalization"
        else:
            rank = self.dotProduct
            method = "Dot product (default method)"

        for doc in self.corpus:
            docScore = rank(query, doc)
            if docScore >= bestScore:
                bestScore = docScore
                bestDoc = doc
        print("{} returned document {} with a comparison score of {}".format(method, bestDoc, bestScore))
        return bestDoc
