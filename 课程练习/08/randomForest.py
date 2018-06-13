from random import seed
from random import randrange
from math import log
from math import sqrt
import pandas as pd
import numpy as np

#不放回的随机将dataSet划分成n_folds个数据集，
#并且在此打乱了顺序
def cross_validation_split(dataSet,n_folds):
	dataSet_split=list()
	dataSet_copy=list(dataSet)
	fold_size=int(len(dataSet)/n_folds)
	for i in range(n_folds):
		fold=list()
		
		while(len(fold)<fold_size):
			index=randrange(len(dataSet_copy))
			fold.append(dataSet_copy.pop(index))
		
		dataSet_split.append(fold)
	return dataSet_split



#输入的是实际分类 和 预测分类
#得到的是预测准确率的百分比
def accuracy_metric(actual,predicted):
	correct=0
	for i in range(len(actual)):
		if actual[i]==predicted[i]:
			correct+=1

	return correct/float(len(actual))*100.0
#其中的algorithm是要带入rondom_tree这个函数的
#也就是数据处理是在计算之前处理好了的



#############决策树的内容
def test_split(index,value,dataSet):
	left,right=list(),list()
	for row in dataSet:
		if row[index]<value:
			left.append(row)
		else:
			right.append(row)

	return left,right
	#当groups=test_split(...)时，groups是一个元组，left，right是两个其中元素
	#通过待测特征的值将数据划分为两个部分

###基尼不纯度
def gini_index(groups,class_values):
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
def get_split(dataSet,n_features):
	class_values=list(set(row[-1] for row in dataSet))
	b_index,b_value,b_score,b_groups=999,999,999,None
	features=[]
	#把n_features打乱顺序
	#每次随机挑取n_features个列进行分析
	#也就是每棵树的特征选取都不一样
	while len(features)<n_features:
		index=randrange(len(dataSet[0])-1)
		if index not in features:
			features.append(index)

	##这个位置可以改进的，可以把内层循环去掉相同项
	#并不是想要row 而是想要row[index]这个值
	for index in features:
		for row in dataSet:
			groups=test_split(index,row[index],dataSet)
			gini=gini_index(groups,class_values)
			if gini<b_score:
				b_index,b_value,b_score,b_groups=index,row[index],gini,groups
				#index是特征的编号，row[index]是特征的值
	return {'index':b_index,'value':b_value,'groups':b_groups}


###到达指定情况(树深度)时，把该特征组合归为，标签数量最多的一类
def to_terminal(group):
	outcomes=[row[-1] for row in group]
	return max(set(outcomes),key=outcomes.count)

#利用递归获取树
#多次调用了自己和get_split函数
##树分的时候依旧有原来的特征，但是根据大小将两部分数据分开成左右部分
##有的决策树没有分左右，这样通过值匹配找到类别很难
def split(node,max_depth,min_size,n_features,depth):
	left,right=node['groups']
	###node定义时就一个给出了最优的特征编号 和特征值
	if not left or not right:
		node['left']=node['right']=to_terminal(left+right)
		#如果其中一个为空，此时只计算另外一个就可以

		return
	if depth>=max_depth:
		node['left'],node['right']=to_terminal(left),to_terminal(right)
		return
	if len(left)<=min_size:
		node['left']=to_terminal(left)

	else:
		node['left']=get_split(left,n_features)
		split(node['left'],max_depth,min_size,n_features,depth+1)
	if len(right)<=min_size:
		node['right']=to_terminal(right)
	else:
		node['right']=get_split(right,n_features)
		split(node['right'],max_depth,min_size,n_features,depth+1)


def build_tree(train,max_depth,min_size,n_features):
	root=get_split(train,n_features)
	split(root,max_depth,min_size,n_features,1)
	return root
################################决策树的内容
#通过树node，解码，row属于哪一类
def predict(node,row):
	if row[node['index']]<node['value']:
		if isinstance(node['left'],dict):
			return predict(node['left'],row)
		else:
			return node['left']
	else:
		if isinstance(node['right'],dict):
			return predict(node['right'],row)
		else:
			return node['right']

####按比例随机取一部分点
#在分好块中再随机选取，放回的选(可以重复)
def subSample(dataSet,ratio):
	sample=list()
	n_sample=round(len(dataSet)*ratio)
	while len(sample)<n_sample:
		index=randrange(len(dataSet))
		# index=np.round(np.random.rand()*len(dataSet))
		sample.append(dataSet[index])
	return sample



#返回预测对最多的那棵树的预测值
def bagging_predict(trees,row):
	predictions=[predict(tree,row) for tree in trees]
	return max(set(predictions),key=predictions.count)

def random_forest(train,test,max_depth,min_size,sample_size,n_trees,n_features):
	trees=list()
	for i in range(n_trees):
		sample=subSample(train,sample_size)
		##sample_size实际是一个比例，生成的新的train数量与原有train数量的比例
		tree=build_tree(sample,max_depth,min_size,n_features)
		trees.append(tree)
	predictions=[bagging_predict(trees,row) for row in test]###随机森林的预测
	return predictions

def evaluate_algorithm(dataSet,algorithm,n_fold,*args):
	folds=cross_validation_split(dataSet,n_folds)
	#folds是随机排列，切割好，分好块的数据
	scores=list()
	for fold in folds:
		train_set=list(folds)
		train_set.remove(fold)
		train_set=sum(train_set,[])
		#sum可以合并两个list,本来train_set是分开的多个folds,现在合并到一起
		##trian_set去掉了fold 
		##fold 用于检测
		#这样减小了分类和检测具有相关性
		################################
		###把验证集的标签去掉，单独放在actual中
		test_set=list()
		for row in fold:
			row_copy=list(row)
			test_set.append(row_copy)
			row_copy[-1]=None
		actual=[row[-1] for row in fold]
		###把验证集的标签去掉，单独放在actual中
		predicted=algorithm(train_set,test_set,*args)
		#*args表示0个或者多个位置参数
		accuracy=accuracy_metric(actual,predicted)
		scores.append(accuracy)
	return scores


seed(1)
filename='sonar.all-data.csv'
df=pd.read_csv(filename,header=None,index_col=None)
data=df.values
np.random.shuffle(data)
print(data)
def str_column_to_int(dataSet,column):
	class_values=[row[column] for row in dataSet]
	unique=set(class_values)
	lookup=dict()
	for key,value in enumerate(unique):
		lookup[value]=key
	for row in dataSet:
		row[column]=lookup[row[column]]
	return lookup
str_column_to_int(data,-1)
data=data.tolist()

n_folds=5
max_depth=10
min_size=3
sample_size=1.5
#每一折中，每一棵树的数据量与该折数据量的比例
#等于1的话就是每棵树的数据量与该折相同，但是顺序打乱了
# n_features=int(len(dataSet[0])-1)
n_features=int(sqrt(len(data[0])-1))
#n_features每个树中最多对n_features个特征分类


for n_trees in range(1,11,4):
	scores=evaluate_algorithm(data,random_forest,n_folds,\
		max_depth,min_size,sample_size,n_trees,n_features)

	print('Trees: %d'%n_trees)
	print('Scores: %s'%scores)
	print('Mean Accuracy: %.3f%%'%(sum(scores)/float(len(scores))))
	print('Max Accuracy: %.3f%%'%(max(scores)))


