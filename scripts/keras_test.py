import numpy
import pandas
import scripts.read_csv as read_csv
import scripts.doc2vec_test as doc2vec_test
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


seed = 7
numpy.random.seed(seed)

dataset = read_csv.return_dataset()
doc2vec_transformer = doc2vec_test.doc2vec_converter_object()
X = []
Y = []
for i in range(5):
    X.append(doc2vec_transformer.trans(dataset[i][0]))
    Y.append(dataset[i][1])

# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(60, input_dim=300, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# evaluate model with standardized dataset
estimators = []
estimators.append(('standarized', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline(), epochs=3, batch_size=5, verbose=0)))
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))