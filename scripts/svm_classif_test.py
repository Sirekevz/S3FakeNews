# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 08:59:39 2019

@author: Kefer.Kathrin-Maria
"""

import matplotlib.pyplot as plt
from gensim.models import Doc2Vec
from sklearn import datasets, svm, metrics
import scripts.read_csv as read_csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import time
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.model_selection import KFold
from scripts import doc2vec_test

# Calculate feature vectors
print("Starting")
dataset = read_csv.return_dataset()
doc2vec_transformer = doc2vec_test.doc2vec_converter_object()



list_of_docs = list(np.array(dataset)[:, 0])
list_of_labels = list(np.array(dataset)[:, 1])

#division into positive and negative examples
#indx_of_pos = np.where(np.array(dataset)[:, 1] > 0)
#indx_of_neg = np.where(np.array(dataset)[:, 1] == 0)
#docs_positive = list(np.array(dataset[indx_of_pos, 0]))
#docs_negative = list(np.array(dataset[indx_of_neg, 0]))


list_of_vectors = [doc2vec_transformer.trans(data) for data in list_of_docs]

kf = KFold(n_splits=5, shuffle=True)
#kf.split(list_of_vectors)

for train_index, test_index in kf.split(list_of_vectors):
    print("TRAIN:", train_index, "TEST:", test_index)

    classifiers = [
        ('LinearSVC', OneVsRestClassifier(LinearSVC(random_state=42))),
        ('Decision Tree', DecisionTreeClassifier(max_depth=5)),
        ('Random Forest (50 estimators)',
         RandomForestClassifier(n_estimators=50, n_jobs=10)),
        ('Random Forest (200 estimators)',
         RandomForestClassifier(n_estimators=200, n_jobs=10)),
        ('Logistic Regression (C=1)',
         OneVsRestClassifier(LogisticRegression(C=1))),
        ('Logistic Regression (C=1000)',
         OneVsRestClassifier(LogisticRegression(C=10000))),
        ('k nn 3', KNeighborsClassifier(3)),
        ('k nn 5', KNeighborsClassifier(5)),
        ('Naive Bayes', OneVsRestClassifier(GaussianNB())),
        ('SVM, linear', OneVsRestClassifier(SVC(kernel="linear",
                                                C=0.025,
                                                cache_size=200))),
        ('SVM, adj.', OneVsRestClassifier(SVC(probability=False,
                                              kernel="rbf",
                                              C=2.8,
                                              gamma=.0073,
                                              cache_size=200))),
        # ('AdaBoost', OneVsRestClassifier(AdaBoostClassifier())),  # 20 minutes to train
        # ('LDA', OneVsRestClassifier(LinearDiscriminantAnalysis())),  # took more than 6 hours
        ('RBM 100', Pipeline(steps=[('rbm', BernoulliRBM(n_components=100)),
                                    ('logistic', LogisticRegression(C=1))])),
        # ('RBM 100, n_iter=20',
        #  Pipeline(steps=[('rbm', BernoulliRBM(n_components=100, n_iter=20)),
        #                  ('logistic', LogisticRegression(C=1))])),
        # ('RBM 256', Pipeline(steps=[('rbm', BernoulliRBM(n_components=256)),
        #                             ('logistic', LogisticRegression(C=1))])),
        # ('RBM 512, n_iter=100',
        #  Pipeline(steps=[('rbm', BernoulliRBM(n_components=512, n_iter=10)),
        #                  ('logistic', LogisticRegression(C=1))])),
    ]

    print(("{clf_name:<30}: {score:<5}  in {train_time:>5} /  {test_time}")
          .format(clf_name="Classifier",
                  score="score",
                  train_time="train",
                  test_time="test"))
    print("-" * 70)
    for clf_name, classifier in classifiers:
        t0 = time.time()
        classifier.fit([list_of_vectors[i] for i in train_index], [list_of_labels[i] for i in train_index])
        t1 = time.time()
        # score = classifier.score(xs['test'], ys['test'])
        preds = classifier.predict([list_of_vectors[i] for i in test_index])
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        t2 = time.time()
        # res = get_tptnfpfn(classifier, data)
        # acc = get_accuracy(res)
        # f1 = get_f_score(res)
        acc = accuracy_score(y_true=[list_of_labels[i] for i in test_index], y_pred=preds)
        f1 = fbeta_score(y_true=[list_of_labels[i] for i in test_index], y_pred=preds, beta=1, average="weighted")
        print(("{clf_name:<30}: {acc:0.2f}% {f1:0.2f}% in {train_time:0.2f}s"
               " train / {test_time:0.2f}s test")
              .format(clf_name=clf_name,
                      acc=(acc * 100),
                      f1=(f1 * 100),
                      train_time=t1 - t0,
                      test_time=t2 - t1))
        # print("\tAccuracy={}\tF1={}".format(acc, f1))
    break
# Get classifiers
