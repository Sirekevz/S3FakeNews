import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# load the data
dataset = pd.read_csv("data/train.csv")

# #### PLOT NUMBER OF ARTICLES AND AUTHORS ####
# sizes = []
# for batch in dataset.groupby("author"):
# 	if batch[1].shape[0] < 30:
# 		sizes.append(batch[1].shape[0])
# plt.hist(sizes, bins = 100)
# plt.show()
# ##############################################


# convert into numpy array, only text and label
dataset = np.array(dataset[["text", "label"]])

# Remove \n and trailing spaces. Get rid of missing values.
i = 0
places_with_nan = []
for row in dataset:
    if str(row[0]) == "nan":
        places_with_nan.append(i)
        continue

    row[0] = row[0].replace("\n", " ").strip(" ")
    i += 1
dataset = np.delete(dataset, places_with_nan, axis=0)


#prob not necessary, division into pos and neg examples
dataPos = []
dataNeg = []
for row in dataset:
    if (row[1] == 1):
        dataPos.append(row[0])
    else:
        dataNeg.append(row[0])

#taken from https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5

tagged_data = [TaggedDocument(words=word_tokenize(str(_d)), tags=[str(i)])
for i, _d in enumerate(dataPos)]

max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print(
        'iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)

# decrease the learning rate
model.alpha -= 0.0002

# fix the learning rate, no decay
model.min_alpha = model.alpha

model.save(
    "d2v.model")
print(
    "Model Saved")


