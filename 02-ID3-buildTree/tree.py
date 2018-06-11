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

	dataSet = [[u'青年', u'否', u'否', u'一般', u'拒绝'],
                [u'青年', u'否', u'否', u'好', u'拒绝'],
                [u'青年', u'是', u'否', u'好', u'同意'],
                [u'青年', u'是', u'是', u'一般', u'同意'],
                [u'青年', u'否', u'否', u'一般', u'拒绝'],
                [u'中年', u'否', u'否', u'一般', u'拒绝'],
                [u'中年', u'否', u'否', u'好', u'拒绝'],
                [u'中年', u'是', u'是', u'好', u'同意'],
                [u'中年', u'否', u'是', u'非常好', u'同意'],
                [u'中年', u'否', u'是', u'非常好', u'同意'],
                [u'老年', u'否', u'是', u'非常好', u'同意'],
                [u'老年', u'否', u'是', u'好', u'同意'],
                [u'老年', u'是', u'否', u'好', u'同意'],
                [u'老年', u'是', u'否', u'非常好', u'同意'],
                [u'老年', u'否', u'否', u'一般', u'拒绝'],
                ]
	feature_index = [u'年龄', u'有工作', u'有房子', u'信贷情况']

	# dataSet=[[1,1,'yes'],
	# 	[1,1,'yes'],
	# 	[1,0,'no'],
	# 	[0,1,'no'],
	# 	[0,1,'no'],
	# 	]
	# feature_index=['0:no surfacing','1:flippers']
	
	
	return dataSet,feature_index

def splitDataSet(dataSet,axis,value):
	retDataSet=[]
	for featVec in dataSet:
		if (featVec[axis]==value):
			# reducedFeatVec=featVec[:axis]
			#numpy中没有extend,而是concatenate
			# reducedFeatVec=np.concatenate((reducedFeatVec,featVec[axis+1:]))
			# reducedFeatVec.extend(featVec[axis+1:])
			reducedFeatVec=featVec.copy()
			del(reducedFeatVec[axis])
			retDataSet.append(reducedFeatVec)
			# retDataSet.append(featVec)
	return retDataSet


def chooseBestFeatureToSplit(dataSet):
	numFeatures=len(dataSet[0])-1
	baseInfoGain=calcShannonEnt(dataSet)
	bestInfoGain=0.0;bestFeature=-1
	for i in range(numFeatures):

		featureList=[example[i] for example in dataSet]
		uniqueFeatures=set(featureList)
		currentEnt=0.0

		for feature in uniqueFeatures:
			subDataSet=splitDataSet(dataSet,i,feature)
			prob=float(len(subDataSet))/len(dataSet)
			currentEnt+=prob*calcShannonEnt(subDataSet)
		newInfoGain=baseInfoGain-currentEnt
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


def creatTree(dataSet,featureList):
	classList=[example[-1] for example in dataSet]
	if classList.count(classList[0])==len(classList):
		
		return classList[0]
	if len(dataSet[0])==1:
		return majorityCnt(classList)
		
	bestFeature=chooseBestFeatureToSplit(dataSet)

	##更新bestFeatLabel 和 featureList
	bestFeatLabel=featureList[bestFeature]
	del(featureList[bestFeature])
	##

	##这个是推荐使用的方法；为CART的树结构理解做准备
	myTree={'feature_index':bestFeatLabel,'child':{}}
	featValues=[example[bestFeature] for example in dataSet]
	uniqueFeat=set(featValues)
	myTree['feature_index']=bestFeatLabel
	for value in uniqueFeat:
		myTree['child'][value]=splitDataSet(dataSet,bestFeature,value)
		myTree['child'][value]=creatTree(myTree['child'][value],featureList)
	
	##下面的写法可以使字典的结构简单化，但是不推荐；
	# myTree={bestFeatLabel:{}}
	# featValues=[example[bestFeature] for example in dataSet]
	# uniqueFeat=set(featValues)
	# for value in uniqueFeat:
	# 	myTree[bestFeatLabel][value]=splitDataSet(dataSet,bestFeature,value)
	# 	myTree[bestFeatLabel][value]=creatTree(myTree[bestFeatLabel][value],featureList)

	return myTree

def getLeaf(tree,test):
	index=tree['feature_index']
	value=test[index]
	# if ~isinstance(tree['child'][value],dict):
	if type(tree['child'][value]).__name__ == 'dict':
		
		tree=tree['child'][value]
		getLeaf(tree, test)
	return tree['child'][value]

	
	
	



def main():
	dataSet,fea=createDataSet()
	Ent=calcShannonEnt(dataSet)
	print(Ent)
	b_feature=chooseBestFeatureToSplit(dataSet)
	print(fea[b_feature])
	fea=[0,1,2,3]
	myTree1=creatTree(dataSet,fea)
	print(myTree1)
	print(getLeaf(myTree1,[u'老年', u'否', u'否', u'一般', u'拒绝']))
	
	print(getLeaf(myTree1,[u'老年', u'否', u'否', u'一般', u'拒绝']))
main()



