# S3FakeNews
Project for the SIGEVO Summer School 2019

.csv files should be located in /data/

============
Tentative overview/summary (10h00)
(can also be a plan for the presentation :-)

Goal: try to discriminate fake news from true news by their content
Mean: design features (different flavours), and apply classification
Data: labeled set of documents

Milestones
1- design features W using word2vec (Adam, Agurren, Evzen)
2- design features T using topics + stats on word counts, e.g., tfidf  (Kathrin, Lucia)
3- agree on a common API (e.g., scikitlearn?) and a choice of classifiers 
5- design an experimental agenda, e.g. 
   * compare results obtained using only W, only T, and W+T
   * study sensitivity of results w.r.t. number of features in W and T (as both approaches should allow to choose the number of features you want to create (word2vec) or keep (tfidf)
   * in particular depending on the chosen classifier (this is where an evolutionary classifier might give better results than, say, SVM when going large scale).
5- run first experiments 
6- prepare the 10-12 slides presentation (can be done in parallel with the rest) 
7- 15h30: beginning of presentations
