import read_csv
import doc2vec_test


dataset = read_csv.return_dataset()
doc2vec_transformer = doc2vec_test.doc2vec_converter_object()



index = 5
print("vector",doc2vec_transformer.trans(dataset[index][0]))
print("class", dataset[index][1])
