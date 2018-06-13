
# import numpy as np
# def data_split(data,index,value,lr):
# 	data=np.array(data)
# 	m,n=data.shape
# 	pre=np.ones((m,1))
# 	# for i in range(m):
# 	# 	if data[i,index]<value:
# 	# 		if lr=='l':
# 	# 			pre[i]=-1
# 	# 	if data[i,index]>=value:
# 	# 		if lr=='r':
# 	# 			pre[i]==-1
# 	# return pre
# 	if lr=='l':
# 		pre[data[:,index]<value]=-1
# 	else:
# 		pre[data[:,index]>=value]=-1
# 	return pre

# # data=[[1,2,3],[2,3,4],[3,4,5]]
# # print(data_split(data,1,3,'l'))

# #通过加权错误率计算最优的决策树桩
# #data是矩阵，label是m行1列的array,w也是m行1列的array
# def tree_1(data,label,w):
# 	data=np.array(data)
# 	m,n=data.shape

# 	min_err=999
# 	b_pred=np.zeros((m,1))
# 	stump={'index':0,'value':0,'lr':'l'}
# 	for index in range(n):
# 		fea=data[:,index]
# 		fea_unique=list(set(fea))
# 		for value in fea_unique:
# 			for lr in ['l','r']:
# 				pred=data_split(data,index,value,lr)
# 				###计算加权错误率
# 				err=np.zeros((m,1))
# 				err[pred!=label]=1
# 				w_err=np.matmul(err.T,w)
# 				if w_err<min_err:
# 					min_err=w_err
# 					b_pred=pred
# 					stump['index']=index;stump['value']=value;stump['lr']=lr
# 	return min_err,b_pred,stump


# def adaboost(data,label,n_trees):
# 	data=np.array(data)
# 	label=np.array(label)
# 	m,n=data.shape
# 	w=np.ones((m,1))/m
	
# 	adaboost_trees={}

# 	pred=np.ones((m,1))  #为了检测当前预测效果加的
# 	for i in range(n_trees):
# 		min_err,b_pred,stump=tree_1(data, label, w)

# 		alpha=0.5*np.log((1-min_err)/max(min_err,1e-18))
# 		# z=np.matmul(w.T,np.exp(-alpha*np.multiply(label ,b_pred)))
# 		# w=np.multiply(w,np.exp(-alpha*np.multiply(label,b_pred)))/z
# 		w=np.multiply(w,np.exp(-alpha*np.multiply(label,b_pred)))
# 		w=w/np.sum(w)
		
# 		adaboost_trees[i]=(alpha,stump)
# 		###总体流程已经结束，下面是监控每一步的结果
# 		print(stump)
# 		pred+=alpha*b_pred
# 		predict=np.ones((m,1))
# 		predict[pred<0]=-1
# 		#打印当前的正确率
# 		print((predict==label).reshape(-1).tolist().count(True)/m)
# 		# print((predict==label).flatten().tolist().count(True)/m)
# 		###总体流程已经结束，下面是监控每一步的结果
# 	return adaboost_trees









# import pandas as pd
# df=pd.read_csv('sonar-mine.csv',header=None)
# df[df.columns[-1]]=df[df.columns[-1]].map({'M':-1,'R':1})


# dataSet=df.values


# np.random.shuffle(dataSet);
# test=dataSet[-30:].copy()
# dataSet=dataSet[:-30].copy()

# print(dataSet.shape)
# data=np.array([d[:-1] for d in dataSet])
# label=np.array([d[-1] for d in dataSet])
# test_data=np.array([d[:-1] for d in test])
# test_label=np.array([d[-1] for d in test])

# fx=adaboost(data,label.reshape((-1,1)),20)
# print(fx)



def split_data(data,index,value):
	m=len(data)
	left=[];right=[]
	for d in data:
		if data[index]<value:
			left.append(d)
		else:
			right.append(d)
	return left , right


def calc_gini(data):
	gini=0.0
	label=[d[-1] for d in data]
	for l in set(label):
		p=label.count(l)/len(label)
		gini+=p*(1-p)
	return gini
def cond_gini(groups):
	m=len(groups[0]);m+=len(groups[1])
	gini=0.0
	for group in groups:
		gini+=len(group)/m*calc_gini(group)
	return gini
def get_split(data):
	b_gini=1.0;
	b_index=-1;b_value=999;

	for index in range(len(data[0])-1):
		fea=[d[index] for d in data]
		fea_space=set(fea)
		for value in fea_space:
			groups=split_data(data, index, value)
			gini=cond_gini(groups) 

			if gini<b_gini:
				b_gini=gini
				b_index=index;b_value=value;b_groups=groups
	return b_index,b_value,b_groups


def to_leaf(label):
	return(max(set(label),keys=label.count))

def cart(data)





