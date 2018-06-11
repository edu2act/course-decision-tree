



# def calc_gini(data):
# 	label_list=[d[-1] for d in data]
# 	# label_list=data[:][-1]
# 	label_dict={}
# 	for label in label_list:
# 		label_dict[label]=label_dict.get(label,0)+1
# 	gini=0.0;N=len(data)
# 	print(label_dict.values())
# 	for value in label_dict.values():
# 		gini+=value/N*(1-value/N)
# 	return gini
# print(calc_gini(d))


# def condition_gini(group):
# 	left,right=group
# 	total_len=len(left)+len(right)
# 	cond_gini=0.0
# 	for data in group:
# 		cond_gini+=len(data)/total_len*calc_gini(data)
# 	return cond_gini

def condition_gini(group):
	left,right=group
	total_len=len(left)+len(right)
	cond_gini=0.0
	for data in group:
		N=len(data)
		label_list=[d[-1] for d in data]
		for label in set(label_list):
			pi=label_list.count(label)/N
			cond_gini+=N/total_len*(pi)*(1-pi)		
	return cond_gini

def split_data(data,index,value):
	left,right=[],[]
	for d in data:
		if d[index]<value:
			left.append(d)
		else:
			right.append(d)
	return left,right

def get_split(data):

	b_gini=999
	b_index=-1;b_value=-1;b_groups=[]
	for index in range(len(data[0])-1):
		feature_space=[d[index] for d in data]
		feature_space=set(feature_space)
		for value in feature_space:
			groups=split_data(data, index, value)
			cond_gini=condition_gini(groups)
			if cond_gini<b_gini:
				b_gini=cond_gini
				b_index,b_value=index,value
				b_groups=groups
	return b_index,b_value,b_groups

dataSet=[[1,1,'yes'],
	[1,1,'yes'],
	[1,0,'no'],
	[0,1,'no'],
	[0,1,'no']]
print(get_split(dataSet))

