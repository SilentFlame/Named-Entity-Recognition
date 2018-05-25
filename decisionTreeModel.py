import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


X = pd.read_csv('featureVector.csv')

y = X['word.Tag']

# removing the Tag column from X to keep it as feature only.
X.drop('word.Tag', axis=1, inplace=True)

# handelling the NaN and inf values in the dataset
X=X.astype('float32')
y=y.astype('float32')
X = np.nan_to_num(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

dtc = DecisionTreeClassifier(max_depth=32, class_weight=[{0:1,1:1}, {0:1,1:50}, {0:1,1:18},{0:1,1:1940}, {0:1,1:70},{0:1,1:3},{0:1,1:25}])
gnb = GaussianNB()
clf = RandomForestClassifier(max_depth=10)

# fit
dtc.fit(X_train, y_train)
gnb.fit(X_train, y_train)
clf.fit(X_train, y_train)

# predict
y_pred = dtc.predict(X_test)
target_names = ['I-Loc', 'B-Org', 'I-Per', 'Other', 'B-Per', 'I-Org', 'B-Loc']

# print
print "Results for Decision tree.."

print(classification_report(y_test, y_pred, target_names=target_names))


# f1 score
score = f1_score(y_pred, y_test, average='weighted')
print "Decision Tree F1 score: {:.2f}".format(score)


print "Results for Naive Bayes..."
y_pred = gnb.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))

# f1 score
score = f1_score(y_pred, y_test, average='weighted')
print "Naive Bayes F1 score: {:.2f}".format(score)



print "Results for Random Forest..."
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))

# f1 score
score = f1_score(y_pred, y_test, average='weighted')
print "random Forest F1 score: {:.2f}".format(score)

# # Cross validation on Data
# pred = cross_val_predict(estimator=dtc, X=X, y=y, cv=5)
# print(classification_report(pred, y, target_names))