from numpy import *
import numpy as np
from math import log

def split(data,index,value,lr):
	m,n=data.shape
	pre=ones((m,1))

	if (lr=='l'):
		pre[data[:,index]<value]=-1
	else:
		pre[data[:,index]>=value]=-1
	return pre

def tree1(data,labelList,w):
	data=np.mat(data);
	m,n=data.shape
	min_err=999
	b_pre=np.ones((m,1))
	stump={'index':0,'value':0,'lr':'l'}
	for index in range(n):
		values=data[:,index]
		values=set([v[0] for v in values.tolist()])
		# values=set(values)
		for value in values:
			for lr in ['l','r']:
				pre=split(data, index, value, lr)
				err=np.mat(zeros((m,1)))
				err[pre!=labelList]=1
				w_err=err.T*w
				if w_err<min_err:
					min_err=w_err
					b_pre=pre
					stump['index'],stump['value'],stump['lr']=index,value,lr
	return min_err,b_pre,stump


def Adaboost(data,labelList,n_stump):
	data=np.mat(data)
	m,n=data.shape
	w=np.mat(ones((m,1)))/m
	pre=np.mat(zeros((m,1)))
	fx={}
	for i in range(n_stump):
		min_err,b_pre,stump=tree1(data,labelList,w)

		b_index,b_value,b_lr=stump['index'],stump['value'],stump['lr']
		alpha=0.5*log((1-min_err)/(min_err))##被除数最好加一个小变量防止为0
		z=w.T*exp(-alpha*np.multiply(labelList ,b_pre))
		w=np.multiply(w/(z),exp(-alpha*np.multiply(labelList ,b_pre)))##被除数最好加一个小变量防止为0
		# w_next=np.multiply(w,exp(-alpha*np.multiply(labelList,b_pre)))
		# w=w_next/np.sum(w_next)

		fx[i]=(alpha,stump)
		print(stump)
		####循环已经完成，下面是为了监控每一次迭代累加之后的预测情况
		pre+=alpha*b_pre
		predict=np.mat(ones((m,1)))
		predict[pre<0]=-1
		# print(labelList)#打印真实标签
		# print(predict)#打印预测标签
		print((predict==labelList).T.tolist()[0].count(True)/len(labelList))#打印正确率
		####循环已经完成，上面是为了监控每一次迭代累加之后的预测情况
	return fx
####例子
# data=mat([[1,1],[1,1],[1,0],[0,1],[0,1]])
# labelList=mat([[1],[1],[-1],[-1],[-1]])
# fx=Adaboost(data,labelList,3)
# print(fx)
####例子


def predict_tree(tree,sample):
	if sample[tree['index']]<tree['value']:
		if tree['lr']=='l':
			pre=-1
		else:pre=1
	else:
		if tree['lr']=='l':
			pre=1
		else:pre=-1
	return pre
####例子
# tree={'index' :1,'value':0,'lr':'l'}
# print(predict_tree(tree, (1,0)))
####例子
def predict(fx,sample):
	pre=0
	# sample.tolist()
	for values in fx.values():
		tree=values[1]
		pre+=values[0]*predict_tree(tree, sample)
	return sign(pre)



import pandas as pd
df=pd.read_csv('sonar-mine.csv',header=None)
df[df.columns[-1]]=df[df.columns[-1]].map({'M':-1,'R':1})


dataSet=df.values



# labels=list(set(labels))
# lab_dict={}
# lab_dict[labels[0]]=-1;lab_dict[labels[1]]=1
# for row in dataSet:
# 	row[-1]=lab_dict[row[-1]]


np.random.shuffle(dataSet);
test=dataSet[-30:].copy()
dataSet=dataSet[:-30].copy()

print(dataSet.shape)
data=np.mat([d[:-1] for d in dataSet])
label=np.mat([d[-1] for d in dataSet])
test_data=np.mat([d[:-1] for d in test])
test_label=np.mat([d[-1] for d in test])

fx=Adaboost(data,label.reshape((-1,1)),20)
print(fx)


pr=map(lambda x: predict(fx,x.T),test_data)
label=np.array(test_label.tolist()[0])
print(label)
predict=np.array([int( i) for i in list(pr)])
print(predict)
print((label==predict).tolist().count(True)/len(label))




