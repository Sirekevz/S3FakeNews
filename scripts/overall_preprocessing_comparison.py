# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 08:59:39 2019
@author: Kefer.Kathrin-Maria
"""

import read_csv_ospath
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import time
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, fbeta_score
import warnings
from pandas import DataFrame
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

topicClassifier=True
doc2vecFeatures=False

# Calculate feature vectors
data_train = read_csv_ospath.return_dataset("train2.csv", True)
data_train_df = DataFrame.from_records(data_train)

X_train, X_test, y_train, y_test = train_test_split(data_train_df.iloc[:,0], data_train_df.iloc[:,1], test_size=0.33, random_state=42)
xs = {'train': X_train, 'test': X_test}
ys = {'train': y_train, 'test': y_test}

# apply the labelling classifier:
if topicClassifier:
    topic_classification_classifier = joblib.load('final_reuter_classification_model.sav')
    
    #TODO: problems with the data format!
    topic_classification_classifier.fit(data_train_df.iloc[0,0], data_train_df.iloc[0,1])

    X_train, X_test, y_train, y_test = train_test_split(data_train_df.iloc[:,0], data_train_df.iloc[:,1], test_size=0.33, random_state=42)
    xs = {'train': data_train_df.iloc[:,0], 'test': data_test_df.iloc[:,0]}
    ys = {'train': data_train_df.iloc[:,1], 'test': data_test_df.iloc[:,1]}
    
    
if doc2vecFeatures:
    #TODO
    

# Get classifiers
classifiers = [
    ('LinearSVC', OneVsRestClassifier(LinearSVC(random_state=42))),
    ('Decision Tree', DecisionTreeClassifier(max_depth=5)),
    ('Random Forest (50 estimators)',
     RandomForestClassifier(n_estimators=50, n_jobs=10)),
    ('Random Forest (200 estimators)',
     RandomForestClassifier(n_estimators=200, n_jobs=10)),
    ('Logistic Regression (C=1)',
     OneVsRestClassifier(LogisticRegression(C=1, solver='saga'))),
    ('Logistic Regression (C=1000)',
     OneVsRestClassifier(LogisticRegression(C=10000, solver='saga'))),
    ('k nn 3', KNeighborsClassifier(3)),
    ('k nn 5', KNeighborsClassifier(5)),
    ('Naive Bayes', OneVsRestClassifier(GaussianNB())),
    ('SVM, linear', OneVsRestClassifier(SVC(kernel="linear",
                                            C=0.025,
                                            cache_size=200)))
]

warnings.filterwarnings('ignore')

print("{clf_name:<30}: {score:<5}  in {train_time:>5} /  {test_time}"
      .format(clf_name="Classifier",
              score="score",
              train_time="train",
              test_time="test"))
print("-" * 70)
for clf_name, classifier in classifiers:
    t0 = time.time()
    classifier.fit(xs['train'], ys['train'])
    t1 = time.time()
    # score = classifier.score(xs['test'], ys['test'])
    preds = classifier.predict(data['x_test'])
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    t2 = time.time()
    # res = get_tptnfpfn(classifier, data)
    # acc = get_accuracy(res)
    # f1 = get_f_score(res)
    acc = accuracy_score(y_true=data['y_test'], y_pred=preds)
    f1 = fbeta_score(y_true=data['y_test'], y_pred=preds, beta=1, average="weighted")
    print(("{clf_name:<30}: {acc:0.2f}% {f1:0.2f}% in {train_time:0.2f}s"
           " train / {test_time:0.2f}s test")
          .format(clf_name=clf_name,
                  acc=(acc * 100),
                  f1=(f1 * 100),
                  train_time=t1 - t0,
                  test_time=t2 - t1))
    # print("\tAccuracy={}\tF1={}".format(acc, f1))

