__author__='mr.y'
'''最简单的决策树实现，使用的是熵和信息增益'''

from math import log
import numpy as np
import operator

log2=lambda x:log(x)/log(2)
def calcShannonEnt(data):
	
	label={}
	for row in data:
		classRow=row[-1]
		# label[classRow]=lable.get(classRow,0)+1
		if label[classRow]==[]:
			label[classRow]=0
		label[classRow]+=1
	shannon=0
	num=len(data)
	for key in label:
		prob=float(label[key])/num
		shannon-=prob*log2(prob)    #log(prob,2)
	return shannon


def calcShannonEnt(dataSet,col=-1):
	numEntries=len(dataSet)
	labelCounts={}
	for featVec in dataSet:
		currentLabel=featVec[col]
		labelCounts[currentLabel]=labelCounts.get(currentLabel,0)+1
	shannonEnt=0.0
	for key in labelCounts:
		prob=float(labelCounts[key])/numEntries
		shannonEnt-=prob*log2(prob)#log(prob,2)
	return shannonEnt



def createDataSet():

	dataSet=[[1,1,'yes'],
		[1,2,'yes'],
		[1,3,'no'],
		[0,4,'no'],
		[0,5,'no'],
		]
	feature_index=['no surfacing','flippers']
	
	feature_index=[i for i in range(3)]
	return dataSet,feature_index

def splitDataSet(dataSet,axis,value):
	retDataSet=[]
	for featVec in dataSet:
		reducedFeature=[]
		if (featVec[axis]==value):
			reducedFeatVec=featVec[0:axis]
			#numpy中没有extend,而是concatenate
			reducedFeatVec=np.concatenate((reducedFeatVec,featVec[axis+1:]))
			# reducedFeatVec.extend(featVec[axis+1:])
			# reducedFeatVec=featVec.copy()
			# del(reducedFeatVec[axis])

			retDataSet.append(reducedFeatVec)
			# retDataSet.append(featVec)
			
	return retDataSet


def chooseBestFeatureToSplit(dataSet):
	numFeatures=len(dataSet[0])-1
	# print(len(dataSet[0]))
	baseInfoGain=calcShannonEnt(dataSet)
	bestInfoGain=0.0;bestFeature=-1
	# bestInfoGain=999
	for i in range(numFeatures):

		featureList=[example[i] for example in dataSet]
		uniqueFeatures=set(featureList)
		currentEnt=0.0
		ent_i=calcShannonEnt(dataSet,i)                            ########+++添加的基于特征i的信息熵
		for feature in uniqueFeatures:
			subDataSet=splitDataSet(dataSet,i,feature)
			prob=float(len(subDataSet))/len(dataSet)
			currentEnt+=prob*calcShannonEnt(subDataSet)
		newInfoGain=(baseInfoGain-currentEnt)/ent_i
		#newInfoGain=baseInfoGain-currentEnt                       ########+++将信息增益改成信息增益率
		# print(newInfoGain,i)
		# if(currentEnt<bestInfoGain):
		if(newInfoGain>bestInfoGain):
			bestInfoGain=newInfoGain
			# bestInfoGain=currentEnt
			bestFeature=i

	return bestFeature


def majorityCnt(classList):
	outcomes=classList.copy()
	return max(set(outcomes),key=outcomes.count)


def creatTree(dataSet,labels,count=0,max_count=4):
	classList=[example[-1] for example in dataSet]
	if classList.count(classList[0])==len(classList):
		
		return classList[0]
	if len(dataSet[0])==1:
		return majorityCnt(classList)
	if count>=max_count:
		return majorityCnt(classList)

	bestFeature=chooseBestFeatureToSplit(dataSet)

	##更新bestFeatLabel 和 labels
	bestFeatLabel=labels[bestFeature]
	
	del(labels[bestFeature])
	label_copy=labels.copy()
	##

	##这个是推荐使用的方法；为CART的树结构理解做准备
	myTree={'feature_index':bestFeatLabel,'child':{}}
	featValues=[example[bestFeature] for example in dataSet]
	uniqueFeat=set(featValues)
	myTree['feature_index']=bestFeatLabel
	for value in uniqueFeat:
		myTree['child'][value]=splitDataSet(dataSet,bestFeature,value)
		myTree['child'][value]=creatTree(myTree['child'][value],label_copy,count+1,max_count)
	
	##下面的写法可以使字典的结构简单化，但是不推荐；
	# myTree={bestFeatLabel:{}}
	# featValues=[example[bestFeature] for example in dataSet]
	# uniqueFeat=set(featValues)
	# for value in uniqueFeat:
	# 	myTree[bestFeatLabel][value]=splitDataSet(dataSet,bestFeature,value)
	# 	myTree[bestFeatLabel][value]=creatTree(myTree[bestFeatLabel][value],labels)

	return myTree

###--------------------添加一个名为data的数据集(离散化了)----------------###
from csv import reader
import split
def load_csv(filename):
	dataSet=list()
	with open(filename,'r') as file:
		csv_reader=reader(file)
		for row in csv_reader:
			if not row :
				continue
			dataSet.append(row)
	return dataSet
# dataSet字符型转换为浮点型
def str_column_to_float(dataSet,column):
	for row in dataSet:
		row[column]=float(row[column].strip())
def str_column_to_int(dataSet,column):
	class_values=[row[column] for row in dataSet]
	unique=set(class_values)
	lookup=dict()
	for key,value in enumerate(unique):
		lookup[value]=key
	for row in dataSet:
		row[column]=lookup[row[column]]
	return lookup
def createD():
	filename='sonar.all-data.csv'
	dataSet=load_csv(filename)

	for i in range(0,len(dataSet[0])-1):
		str_column_to_float(dataSet,i)
	str_column_to_int(dataSet,len(dataSet[0])-1)
	
	return dataSet
###--------------------添加一个名为data的数据集(离散化了)----------------###

def featureSplit(dataSet,method="step",step_num=10):


	if (method=='step'):
		dataSet_copy=np.array(dataSet.copy())
		m,n=dataSet_copy.shape
		for i in range(n-1):
			uniqueFeature=[row[i] for row in dataSet_copy]
			min_num,max_num=min(uniqueFeature),max(uniqueFeature)
			step=(max_num-min_num)/step_num
			
			for j in range(m):
				dataSet_copy[j,i]=((dataSet_copy[j,i]-min_num)/step).astype(np.int32)
	# print(dataSet_copy.shape)	
	return dataSet_copy


import treePlotter2
from pylab import *
def main():
	# dataSet,label=createDataSet()
	dataSet_=createD()
	dataSet=featureSplit(dataSet_,step_num=3)
	label=[i for i in range(len(dataSet[0]))]
	# print(label)
	
	Ent=calcShannonEnt(dataSet)
	# print(Ent)
	feature=chooseBestFeatureToSplit(dataSet)
	# print(feature)
	myTree1=creatTree(dataSet,label)
	# print(myTree1)
	
	mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
	mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
	# 测试决策树的构建
	
	treePlotter2.createPlot(myTree1)

main()



