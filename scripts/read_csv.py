import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def return_dataset():

	#load the data
	dataset = pd.read_csv("../data/train.csv")



	# #### PLOT NUMBER OF ARTICLES AND AUTHORS ####
	# sizes = []
	# for batch in dataset.groupby("author"):
	# 	if batch[1].shape[0] < 30:
	# 		sizes.append(batch[1].shape[0])
	# plt.hist(sizes, bins = 100)
	# plt.show()
	# ##############################################



	# convert into numpy array, only text and label
	dataset = np.array(dataset[["text","label"]]) 

	# Remove \n and trailing spaces. Get rid of missing values.
	i = 0
	places_with_nan = []
	for row in dataset:
		if str(row[0]).strip()=="nan" or isinstance(row[0], float):
			places_with_nan.append(i)
			i += 1
			continue

		row[0] = row[0].replace("\n", " ").strip(" ")
		i += 1
	dataset = np.delete(dataset, places_with_nan, axis=0)
	
	new_data  = []
	for line in dataset:
		text = line[0]
		text = ''.join(ch.lower() for ch in text if ch.isalpha() or ch == " ")
		new_data.append([[word for word in text.split(" ") if len(word)>1],line[-1]])


	
	return new_data

