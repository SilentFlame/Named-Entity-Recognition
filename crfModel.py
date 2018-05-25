import pandas as pd
import numpy as np
from sklearn_crfsuite import CRF
from sklearn.cross_validation import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report



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



data = pd.read_csv("taggedData.csv", encoding="latin1")
data = data.fillna(method="ffill")

getter = SentenceGetter(data)
sentences = getter.sentences

def word2features(sent, i):
    word = sent[i][0]
    
    features = {
        'bias': 1.0,
        'word': word,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.startsWith#()': word.startswith("#"),
        'word.startsWith@()': word.startswith("@"),
        'word.1stUpper()': word[0].isupper(),
        'word.isAlpha()': word.isalpha(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word': word1,
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.startsWith#()': word1.startswith("#"),
            '-1:word.startsWith@()': word1.startswith("@"),
            '-1:word.1stUpper()': word1[0].isupper(),
            '-1:word.isAlpha()': word1.isalpha(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word': word1,
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.startsWith#()': word1.startswith("#"),
            '+1:word.startsWith@()': word1.startswith("@"),
            '+1:word.1stUpper()': word1[0].isupper(),
            '+1:word.isAlpha()': word1.isalpha(),
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]


X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]


crf = CRF(algorithm='l2sgd',
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)


pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
report = flat_classification_report(y_pred=pred, y_true=y)
print(report)

crf.fit(X, y)