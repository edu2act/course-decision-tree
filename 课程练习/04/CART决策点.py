


import pandas as pd
import punning
import numpy as np



dataSet=[[1,1,'yes'],
	[1,1,'yes'],
	[1,0,'no'],
	[0,1,'no'],
	[0,1,'no']]

def split_data(data,index,value):
	left=[];right=[]
	for d in data:
		if (d[index]<value):
			left.append(d)
		else:
			right.append(d)
	return left, right

def condition_gini(groups):
	N=len(groups[0])+len(groups[1])
	cond_gini=0.0
	for group in groups:
		label_list=[d[-1] for d in group]
		label_space=set(label_list)
		n_data=len(label_list)
		for label in label_space:
			pj=label_list.count(label)/n_data
			cond_gini+=n_data/N*pj*(1-pj)
	return cond_gini

def get_split(data):
	fea_num=len(data[0])-1
	min_gini=1;
	b_index=-1;b_value=-1
	b_groups=()
	for i in range(fea_num):
		fea_space=set([d[i] for d in data])
		for value in fea_space:
			groups=split_data(data, i, value)
			cond_gini=condition_gini(groups)
			if cond_gini<min_gini:
				min_gini=cond_gini
				b_index=i;b_value=value
				b_groups=groups
	return b_index,b_value,b_groups

def to_leaf_node(label_list):

	return max(set(label_list),key=label_list.count)

def create_tree(data,max_depth,min_size,depth=0):
	label_list=[d[-1] for d in data]
	if (len(data)<=min_size) or (len(set(label_list))==1) or (depth>=max_depth):		
		return to_leaf_node(label_list)

	index,value,groups=get_split(data)
	tree={'index':index,'value':value,'left':{},'right':{}}

	tree['left']=create_tree(groups[0], max_depth,min_size,depth+1)
	tree['right']=create_tree(groups[1], max_depth,min_size,depth+1)
	tree['groups']=groups
	return tree

def fit(data,max_depth=None,min_size=1):
	if max_depth==None:
		max_depth=100
	return create_tree(data, max_depth, min_size)

def predict(tree,sample):
	index=tree['index'];value=tree['value']
	if sample[index]<value:
		if isinstance(tree['left'],dict):
			 return predict(tree['left'], sample)
		else:
			 return tree['left']
	else:
		if isinstance(tree['right'],dict):
			 return predict(tree['right'], sample)
		else:
			 return tree['right']


import treePlotter_cart
data=pd.read_csv('sonar.csv',header=None)
train_data=data.values.tolist()
np.random.shuffle(train_data)
tree=fit(train_data[:160])

treePlotter_cart.createPlot(tree)
# tree2=punning.prune(tree,train_data[160:])
# # print(tree2)
# treePlotter_cart.createPlot(tree2)

import pruning
tree2=pruning.pruning_CART(tree,train_data[160:])
treePlotter_cart.createPlot(tree2)




# def mean_squar_error(groups):
# 	m_s_e=0.0
# 	for group in groups:
# 		if len(group)==0:
# 			continue
# 		label_list=[d[-1] for d in group]
# 		c=np.array(label_list).mean()
# 		m_s_e+=np.sum(np.power(label_list-c,2))
# 	return m_s_e
