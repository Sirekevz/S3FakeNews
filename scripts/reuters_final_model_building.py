# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:44:08 2019

@author: Kefer.Kathrin-Maria
"""

import reuters_data_loading
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib


"""
Train a classifier for a dataset.
Parameters
----------
categories : list of str
document_ids : list of str
"""
# Calculate feature vectors
data = reuters_data_loading.load_data()

xs = {'train': data['x_train'], 'test': data['x_test']}
ys = {'train': data['y_train'], 'test': data['y_test']}

C_vals=[1,10,100,1000]
#C_vals.extend(list([3**n for n in range(-5,5)]))

parameters_to_tune = {'estimator__C':C_vals, 'estimator__multi_class':['ovr','crammer_singer']}
classifier=GridSearchCV(OneVsRestClassifier(LinearSVC(max_iter=2000)), parameters_to_tune, cv=10, scoring='accuracy')

classifier.fit(xs['train'], ys['train'])

preds = classifier.predict(data['x_test'])

print(classification_report(data['y_test'],preds))
#print(confusion_matrix(data['y_test'], preds))
   
acc = accuracy_score(y_true=data['y_test'], y_pred=preds)
f1 = fbeta_score(y_true=data['y_test'], y_pred=preds, beta=1, average="weighted")
print("Accuracy = " + str(acc * 100))
print("F1 = " + str(f1 * 100))    

print("BEST CLASSIFIER: "+ str(classifier.best_score_))
print("BEST PARAMS: "+ str(classifier.best_params_))

joblib.dump(classifier.best_estimator_, 'final_reuter_classification_model.sav')