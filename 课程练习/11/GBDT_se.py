
# import pandas as pd
# import numpy as np
# from CART_Regression import *



# def GBDT(data1,n_trees):
# 	data_copy=data1.copy()
# 	trees={};label=np.array([])
# 	tree=fit(data_copy,max_depth=3)
# 	trees[0]=tree

# 	predict_label=predict(tree, data_copy)
# 	label=np.concatenate([label,np.array(predict_label)])
# 	print(label[:15])
# 	for j,d in enumerate(data_copy):
# 		d[-1]=d[-1]-predict_label[j]
		
# 	for i in range(1,n_trees):
# 		for j,d in enumerate(data_copy):
# 			d[-1]=d[-1]-predict_label[j]
# 		tree=fit(data_copy,max_depth=3)
# 		trees[i]=tree
# 		predict_label=predict(tree, data_copy)
# 		print(predict_label[:15])
# 		label=np.concatenate([label,0.01*(np.array(predict_label))])
# 		print(np.sum(label.reshape((i+1,-1)),axis=0)[:15])
# 	# tree=fit(data_copy,max_depth=2)
# 	# pred_label=predict(tree,data_copy)
# 	# print(pred_label[:5])
# 	return trees



# filename='sonar-mine.csv'
# data=pd.read_csv(filename)
# from random import shuffle
# (data[data.columns[-1]])=(data[data.columns[-1]]).map({"M":0,"R":1})
# data=data.values
# data=data.tolist()
# shuffle(data)

# data_array=np.array(data)
# GBDT(data,3)
# print([d[-1] for d in data_array[:15]])

import numpy as np
from CART_Regression import *


def GBDT(data,n_trees,eta=0.1):
	trees={}
	label=np.array([])
	tree=fit(data,max_depth=3)
	trees[0]=(1,tree)
	data_features=[d[:-1] for d in data]

	pred_label=predict(tree, data_features)
	print(pred_label[:30])
	label=np.concatenate([label,np.array(pred_label)])
	for i in range(1,n_trees):
		
		for j,d in enumerate(data):
			d[-1]=(d[-1]-pred_label[j])
		# for d_fea,d in zip(data_features,data):
			
		tree=fit(data,max_depth=3)
		trees[i]=(eta,tree)


		pred_label=predict(tree, data_features)
		label=np.concatenate([label,eta*np.array(pred_label)])
		print(np.sum(label.reshape((i+1,-1)),axis=0)[:30])
	return trees

import pandas as pd
from random import shuffle
data=pd.read_csv('sonar-mine.csv',header=None)
(data[data.columns[-1]])=(data[data.columns[-1]]).map({"M":0,"R":1})

data=data.values
data1=data.tolist()
shuffle(data1)
print([d[-1] for d in data1[:30]])
trees=GBDT(data1,10)



