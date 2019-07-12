import read_csv
import doc2vec_test
from keras.models import Sequential
from keras.layers import Dense

import numpy as np
from tqdm import tqdm


dataset = read_csv.return_dataset()
doc2vec_transformer = doc2vec_test.doc2vec_converter_object()



index = 5
print("vector",doc2vec_transformer.trans(dataset[index][0]))
print("class", dataset[index][1])


## Model ##

model = Sequential()
model.add(Dense(units=5, activation='tanh', input_dim=10))
model.add(Dense(units=5, activation='tanh'))
model.add(Dense(units=3, activation='tanh'))
model.add(Dense(units=1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



N = 20761
train_perc = 0.75

print("transformig with doc2vec")
x_train = np.array([doc2vec_transformer.trans(dataset[index][0]) for index in tqdm(range(int(N * train_perc)))])
y_train = np.array([dataset[index][1] for index in range(int(N*train_perc))])



model.fit(x_train, y_train, epochs=20, batch_size=32)


print("generating test set")
x_test = np.array([doc2vec_transformer.trans(dataset[index][0]) for index in tqdm(range(int(N * train_perc),N))])
y_test = np.array(np.array([dataset[index][1] for index in range(int(N*train_perc),N)]))

classes = model.predict(x_test, batch_size=128)


classes = np.around(classes, 0)

print("error perc: ",100*np.average(np.absolute(classes - y_test)))
print(classes)