import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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
	if str(row[0])=="nan":
		places_with_nan.append(i)
		continue

	row[0] = row[0].replace("\n", " ").strip(" ")
	i += 1
dataset = np.delete(dataset, places_with_nan, axis=0)

# print first 10 cases.
print(dataset[0:10])

