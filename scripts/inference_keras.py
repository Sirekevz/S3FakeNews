import read_csv
import doc2vec_test


dataset = read_csv.return_dataset()
doc2vec_transformer = doc2vec_test.doc2vec_converter_object()


print(dataset[0:5])

