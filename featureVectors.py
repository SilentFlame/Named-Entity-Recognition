import pandas as pd
import numpy as np
import csv
import sys
import math



reload(sys)
sys.setdefaultencoding('utf8')

class SentenceGetter(object):
            
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sent").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None




def numericFeatures():
    data = pd.read_csv("taggedData.csv", encoding="latin1")

    data = data.fillna(method="ffill")

    words = list(set(data["Word"].values))
    words.append("ENDPAD")
    tags = list(set(data["Tag"].values))

    print tags



    max_len = 50
    word2idx = {w: i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    word2Suff2idx = {w[-2:]: i for i, w in enumerate(words)}
    word3Suff2idx = {w[-3:]: i for i, w in enumerate(words)}
    wordLower2idx = {w.lower(): i for i, w in enumerate(words)}
    binaryIdx = {"True": 1, "False": 0}

    # print binaryIdx[str("False")]

    # X = [[binaryIdx[str(w[5]] for w in s] for s in features]


    getter = SentenceGetter(data)
    # sent = getter.get_next()
    sentences = getter.sentences

    def word2features(sent, i):
        word = sent[i][0]  
        features = {
            'bias': 1.0,
            'word': word2idx[word],
            'word.lower()': wordLower2idx[word.lower()],
            'word[-3:]': word3Suff2idx[word[-3:]],
            'word[-2:]': word2Suff2idx[word[-2:]],
            'word.isupper()': binaryIdx[str(word.isupper())],
            'word.istitle()': binaryIdx[str(word.istitle())],
            'word.isdigit()': binaryIdx[str(word.isdigit())],
            'word.startsWith#()': binaryIdx[str(word.startswith("#"))],
            'word.startsWith@()': binaryIdx[str(word.startswith("@"))],
            'word.1stUpper()': binaryIdx[str(word[0].isupper())],
            'word.isAlpha()': binaryIdx[str(word.isalpha())],
            'word.Tag': tag2idx[sent[i][1]],
        }
        if i > 0:
            word1 = sent[i-1][0]
            features.update({
                '-1:word': word2idx[word1],
                '-1:word.lower()': wordLower2idx[word1.lower()],
                '-1:word.istitle()': binaryIdx[str(word1.istitle())],
                '-1:word.isupper()': binaryIdx[str(word1.isupper())],
                '-1:word.istitle()': binaryIdx[str(word1.istitle())],
                '-1:word.isdigit()': binaryIdx[str(word1.isdigit())],
                '-1:word.startsWith#()': binaryIdx[str(word1.startswith("#"))],
                '-1:word.startsWith@()': binaryIdx[str(word1.startswith("@"))],
                '-1:word.1stUpper()': binaryIdx[str(word1[0].isupper())],
                '-1:word.isAlpha()': binaryIdx[str(word1.isalpha())],
            })
        else:
            features['BOS'] = binaryIdx[str("True")]

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            features.update({
                '+1:word': word2idx[word1],
                '+1:word.lower()': wordLower2idx[word1.lower()],
                '+1:word.istitle()': binaryIdx[str(word1.istitle())],
                '+1:word.isupper()': binaryIdx[str(word1.isupper())],
                '+1:word.istitle()': binaryIdx[str(word1.istitle())],
                '+1:word.isdigit()': binaryIdx[str(word1.isdigit())],
                '+1:word.startsWith#()': binaryIdx[str(word1.startswith("#"))],
                '+1:word.startsWith@()': binaryIdx[str(word1.startswith("@"))],
                '+1:word.1stUpper()': binaryIdx[str(word1[0].isupper())],
                '+1:word.isAlpha()': binaryIdx[str(word1.isalpha())],
            })
        else:
            features['EOS'] = binaryIdx[str("True")]

        return features


    def sent2features(sent):
        return [word2features(sent, i) for i in range(len(sent))]

    X = [sent2features(s) for s in sentences]
    # print features

    # print features
    csv_columns = ['+1:word', '+1:word.1stUpper()', '+1:word.isAlpha()', '+1:word.isdigit()', '+1:word.istitle()','+1:word.isupper()', '+1:word.lower()', '+1:word.startsWith#()', '+1:word.startsWith@()', 'BOS', '-1:word', '-1:word.1stUpper()', '-1:word.isAlpha()', '-1:word.isdigit()', '-1:word.istitle()', '-1:word.isupper()','-1:word.lower()', '-1:word.startsWith#()', '-1:word.startsWith@()', 'EOS', 'bias', 'word', 'word.1stUpper()', 'word.isAlpha()', 'word.isdigit()', 'word.istitle()','word.isupper()', 'word.lower()', 'word.startsWith#()', 'word.startsWith@()', 'word[-2:]', 'word[-3:]', 'word.Tag']
    # print len(csv_columns)

    with open('featureVector.csv', 'wb') as ofile:
    	writer = csv.DictWriter(ofile, csv_columns)
    	writer.writeheader()
    	for s in X:
    		for d in s:
    			# print d
    			writer.writerow(d)

