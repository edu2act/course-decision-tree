from random import randrange
from CART_for_RF import *
import numpy as np

def subSample(dataSet,ratio):
	sample=list()
	n_sample=round(len(dataSet)*ratio)
	while(len(sample)<n_sample):
		index=randrange(len(dataSet))
		# sample=np.concatenate(sample,dataSet[index].copy())
		sample.append(dataSet[index].copy())
	return sample


def randomForest(dataSet,max_depth=10,min_size=3,ratio=1,n_trees=5,n_features=None):
	
	trees=list()
	
	for i in range (n_trees):
		tree=CARTClassifier()
		sample=subSample(dataSet, ratio)
		tree.build_tree(sample,n_features)
		trees.append(tree)
		
		# print(tree.node.index)
	return trees

def rf_predict(trees,row):
	#调用predict函数
	predictions=[tree.predict(row) for tree in trees]
	return (max(predictions,key=predictions.count))

import pandas as pd
filename='sonar-mine.csv'
dataSet=pd.read_csv(filename,header=None)
from random import shuffle
# (dataSet[dataSet.columns[-1]]).map({"M":0,"R":1})
dataSet=dataSet.values
dataSet=dataSet.tolist()
# print(dataSet)
shuffle(dataSet)

trees=randomForest(dataSet[:160],ratio=2,n_trees=30,n_features=int(np.sqrt(43)))


prediction=[rf_predict(trees,row[:-1]) for row in dataSet[160:]]
print(prediction)
real=[row[-1] for row in dataSet[160:]]
print(real)
print([p ==r for p,r in zip(prediction,real)])
		
print([p ==r for p,r in zip(prediction,real)].count(True)/len(real))

