import read_csv
import numpy as np
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile



# dataset = read_csv.return_dataset()
# list_of_docs = list(np.array(dataset)[:,0])


# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(list_of_docs)]
# model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=1)


# fname = get_tmpfile("my_doc2vec_model")
# model.save("model1")

model = Doc2Vec.load("model1")

list_of_words = ["donald", "trump", "is", "the", "actual", "president"]

print(model.infer_vector(list_of_words))

print("done")