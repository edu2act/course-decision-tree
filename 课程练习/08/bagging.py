from random import randrange
from CART import * 	
import numpy as np

def subSample(dataSet,ratio=1):
	sample=list()
	n_sample=round(len(dataSet)*ratio)
	while(len(sample)<n_sample):
		index=randrange(len(dataSet))
		# sample=np.concatenate(sample,dataSet[index].copy())
		sample.append(dataSet[index].copy())
	return sample


def bagging(dataSet,max_depth=10,min_size=1,ratio=1,n_trees=5):
	
	trees=list()
	
	for i in range (n_trees):
		tree=CARTClassifier()
		sample=subSample(dataSet, ratio)
		tree.build_tree(sample)
		trees.append(tree)
		
		# print(tree.node.index)
	return trees

def bagging_predict(trees,row):
	#调用predict函数
	predictions=[tree.predict(row) for tree in trees]
	return (max(predictions,key=predictions.count))

# dataSet=[[1,2,3,1],[1,2,3,1],[2,1,3,2],[2,1,3,2],[2,1,3,2],[2,1,3,2],[1,2,3,1],[1,2,3,1],]

# trees=bagging(dataSet)
# predict=bagging_predict(trees,[2,1,3])
# print(predict)

import pandas as pd
filename='sonar-mine.csv'
data=pd.read_csv(filename)
from random import shuffle
# (dataSet[dataSet.columns[-1]]).map({"M":0,"R":1})
data=data.values
data=data.tolist()
shuffle(data)

trees=bagging(data[:160])



predict=[bagging_predict(trees,row[:-1]) for row in data[160:]]
print(predict)
real=[row[-1] for row in data[160:]]
print(real)
		


