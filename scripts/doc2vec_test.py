import scripts.read_csv as read_csv
import numpy as np
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile


# ### GENERATING THE MODEL ############
# dataset = read_csv.return_dataset()
# list_of_docs = list(np.array(dataset)[:,0])
# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(list_of_docs)]
# model = Doc2Vec(documents, vector_size=10, window=2, min_count=1, workers=1)
# fname = get_tmpfile("my_doc2vec_model")
# model.save("model_size_10")
# ########################################



### LOADING THE MODEL ############
class doc2vec_converter_object:

	def __init__(self):
		print("loading model")
		self._model = Doc2Vec.load("model_size_10")

	def trans(self, list_of_words):
		return self._model.infer_vector(list_of_words)

########################################


