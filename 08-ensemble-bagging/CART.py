#-*-coding:utf-8 -*-
__author__='mr.y'


'''使用的是第二种递归方法'''

from random import randrange
from csv import reader
from math import log
from math import sqrt
import pandas as pd

class Node:
	def __init__(self,index=-1,values=999,left=-1,right=-1):
		self.index=index
		self.values=values
		
		self.left=left
		self.right=right
###创建了一个类为了保存树构造过程中所需的内容
#包括特征索引index,特征最优取值values,保存数据groups,
#左分支left  和 右分支right


class CARTClassifier:
	def __init__(self,max_depth=10,min_size=3,node=Node()):
		self.node=node
		self.max_depth=max_depth
		self.min_size=min_size


	#############决策树的内容
	def test_split(self,index,value,dataSet):
		left,right=list(),list()
		for row in dataSet:
			if row[index]<value:
				left.append(row)
			else:
				right.append(row)

		return left,right
		#当groups=self.test_self.split(...)时，groups是一个元组，left，right是两个其中元素
		#通过待测特征的值将数据划分为两个部分

	###基尼不纯度
	def gini_index(self,groups,class_values):
		gini=0.0
		total_size=float(len(groups[0])+len(groups[1]))
		for class_value in class_values:
			for group in groups:
				size=len(group)
				if size==0:
					continue
				proportion=[row[-1] for row in group].count(class_value)/float(size)
				#proportion表示正确划分的比例
				gini+=float(size)/total_size*(proportion*(1.0-proportion))
				#这的是基尼不纯度
				#如果全部划分正确为0，全部划分不正确也是0
				#
				#不是基于熵的
		return gini

	#总体的数据集，和特征集，最终得到最优的其中其中一个特征
	#这个函数中dataSet没有改动
	#为什么不一样呢，是因为这里的测试集时浮点型的，很少有两个值一样，只能通过大小来区分
	def get_split(self,dataSet):

		node_new=Node()
		class_values=list(set(row[-1] for row in dataSet))
		b_index,b_value,b_score,b_groups=999,999,999,None
		

		features=range(0,len(dataSet[0])-1)
		# print(len(features))

		##这个位置可以改进的，可以把内层循环去掉相同项
		#并不是想要row 而是想要row[index]这个值
		for index in features:
			for row in dataSet:
				groups=self.test_split(index,row[index],dataSet)
				gini=self.gini_index(groups,class_values)
				if gini<b_score:
					node_new.index,node_new.values,node_new.left,node_new.right,b_score=index,row[index],groups[0],groups[1],gini
					#index是特征的编号，row[index]是特征的值
		return node_new


	###到达指定情况(树深度)时，把该特征组合归为，标签数量最多的一类
	def to_terminal(self,group):
		outcomes=[row[-1] for row in group]
		return max(set(outcomes),key=outcomes.count)


	#利用递归获取树
	#多次调用了自己和self.get_split函数
	##树分的时候依旧有原来的特征，但是根据大小将两部分数据分开成左右部分
	##有的决策树没有分左右，这样通过值匹配找到类别很难
	def split(self,node,depth):
		left,right=node.left,node.right
		###node定义时就一个给出了最优的特征编号 和特征值
		if not left or not right:
			node.left=node.right=self.to_terminal(left+right)
			#如果其中一个为空，此时只计算另外一个就可以

			return
		if depth>=self.max_depth:
			node.left,node.right=self.to_terminal(left),self.to_terminal(right)
			return
		if len(left)<=self.min_size:
			node.left=self.to_terminal(left)
		else:
			node.left=self.get_split(left)
			self.split(node.left,depth+1)


		if len(right)<=self.min_size:
			node.right=self.to_terminal(right)
			
		else:
			node.right=self.get_split(right)
			self.split(node.right,depth+1)


	def build_tree(self,Train_Data):
		node=Node()
		node=self.get_split(Train_Data)
		self.split(node,1)
		self.node=node
		# return node
	##########其中的判定条件dict 一定要变成Node

	#输入的是实际分类 和 预测分类
	#得到的是预测准确率的百分比
	def accuracy_metric(self,actual,Validation_Data):
		correct=0
		for i in range(len(actual)):
			if actual[i]==self.predict(Validation_Data[i]):
				correct+=1

		return correct/float(len(actual))*100.0
	def _predict(self,node,row):

		if row[node.index]<node.values:
			
			if isinstance(node.left,Node):
				
				return self._predict(node.left,row)
			else:
				return node.left
		else:
			if isinstance(node.right,Node):
				
				return self._predict(node.right,row)
			else:
				return node.right
	def predict(self,row):
		return(self._predict(self.node,row))








# filename='sonar.all-data.csv'
# dataSet=pd.read_csv(filename,header=None)
# dataSet=dataSet.values
# print(dataSet)

# from random import shuffle
# if __name__ == '__main__':
# 	max_depth=10
# 	min_size=3
# 	print(len(dataSet))
# 	shuffle(dataSet)
# 	tree=CARTClassifier(max_depth,min_size)

# 	tree.build_tree(dataSet[0:150])
			
# 	predictions=[row[0:-1] for row in dataSet[150:160]]

# 	realLabels=[row[-1] for row in dataSet[150:160]]
# 	print([tree.predict(row) for row in predictions])
# 	print(realLabels)
# 	current=tree.accuracy_metric(realLabels,predictions)
# 	print(current)







