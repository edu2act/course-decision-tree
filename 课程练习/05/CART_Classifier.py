# -*- coding :UTF-8 -*-
"""CART回归课件总结，第一种递归方法，带着group的"""


def split_data(dataSet,index,value):#数据集划分
	#和ID3处的不同，每次不删列
	#value是作为分割线，而不是等于才取
	left,right=list(),list()
	for row in dataSet:
		if row[index]<=value:
			left.append(row)
		else:
			right.append(row)

	return left,right

def gini(groups,class_values):#基尼系数
	gini_score=0.0
	total_size=float(len(groups[0])+len(groups[1]))
	for class_value in class_values:
		for group in groups:
			size=len(group)
			if size==0:
				continue
			proportion=[row[-1] for row in group].count(class_value)/float(size)
			gini_score+=float(size)/total_size*(proportion*(1.0-proportion))
	return gini_score

def get_split(dataSet):#求最优特征索引和最优二分标准
	class_values=list(set(row[-1] for row in dataSet))
	b_index,b_value,b_score,b_groups=999,999,999,None
	features=range(0,len(dataSet[0])-1)

	for index in features:
		fea_Space=list(set([row[index] for row in dataSet]))
		##这里将每一个值作为二分标准选取方法,可以采用其他方法		
		for row in fea_Space:
			groups=split_data(dataSet,index,row)
			gini_score=gini(groups,class_values)
			if gini_score<b_score:
				b_index,b_value,b_score,b_groups=index,row,gini_score,groups
	return b_index,b_value,b_groups

def toLeafNode(outcomes):#决策点变成叶子节点
	# outcomes=[row[-1] for row in group]
	return max(set(outcomes),key=outcomes.count)

def createTree(data,max_depth,minsize,depth):#递归建树
	labelList=[d[-1]  for d in data]
	if (len(set(labelList))==1) or len(data)<=minsize or depth>=max_depth:
		return toLeafNode(labelList)

	node=get_split(data)
	left,right=node[2][0],node[2][1]

	tree={'index':node[0],'value':node[1],'left':{},'right':{},'groups':node[2]}
	
	tree['left']=createTree(left,max_depth,minsize,depth+1)
	tree['right']=createTree(right,max_depth,minsize,depth+1)
	return tree	

def fit(train,max_depth=None,minsize=1):#有的递归是第一项区别对待的，同时方便设定参数,因此添加一个函数
	if max_depth==None:
		max_depth=999
	depth=0
	root=createTree(train,max_depth,minsize,depth)
	return root

###predict###predict###predict###predict###predict###predict###predict###predict###predict

def predict(node,row):#使用tree预测单样本row是哪个类别
	index=node['index']
	if row[index]<node['value']:		
		if isinstance(node['left'],dict):
			return predict(node['left'],row)
		else:
			return node['left']
	else:
		if isinstance(node['right'],dict):
			return predict(node['right'],row)
		else:
			return node['right']

def accuracy_metric(actual,predicted):#求预测的正确率，返回的是百分比
	correct=0
	for i in range(len(actual)):
		if actual[i]==predicted[i]:
			correct+=1
	return correct/float(len(actual))*100.0


####可视化
# import treePlotter_cart
# from pylab import *
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题



# def main():
# 	dataSet=[[1,2,'yes'],
# 		[1,2,'yes'],
# 		[1,0,'no'],
# 		[1,1,'no'],
# 		[0,2,'no'],
# 		]
# 	tree=fit(dataSet)
# 	print(tree)
# 	treePlotter_cart.createPlot(tree)

# main()
# #{'index': 1, 'value': 2, 'left': 'no', 'right': {'index': 0, 'value': 1, 'left': 'no', 'right': 'yes'}}


