

from math import log2


def calc_shannon_ent(data):
	num=len(data)
	label_count={}
	for row in data:
		label=row[-1]
		if label not in label_count.keys:
			label_count[label]=0
		label_count[lable]+=1
	shannonEnt=0.0
	for value in label_count.values:
		pi=value/num
		shannonEnt-=pi*log2(pi)
	return


def split_data(data,index,value):
	splitData=list()
	for row in data:
		if row[index]==value:
			del(row[index])
			splitData.append(row)
	return splitData


def get_best_split(data):
	num=len(data)
	feaLen=len(data[0])-1
	baseEnt=calc_shannon_ent(data)
	b_IG=0.0
	for i in range(feaLen):
		feaSpace=set([row[i] for i in data])
		cond_Ent=0.0
		for value in feaSpace:
			subData=split_data(data, i, value)
			subEnt=calc_shannon_ent(subData)


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

	return dataSet,feature_index





