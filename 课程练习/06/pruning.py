


def isTree(obj):
	return isinstance(obj,dict)

def getMean(tree):

	label=[]
	for group in tree['groups']:
		for d in group:
			label.append(d[-1])
	return max(set(label),key=label.count)


def data_split(data,index,value):
	left=[];right=[]
	for d in data:
		if d[index]<value:
			left.append(d)
		else:
			right.append(d)
	return left,right



def pruning_CART(tree,test_data):
	if not test_data:
		return getMean(tree)
	if isTree(tree['left']) or isTree(tree['right']):
		left_data,right_data=data_split(test_data,tree['index'],tree['value'])
		if isTree(tree['left']):
			tree['left']=pruning_CART(tree['left'], left_data)
		if isTree(tree['right']):
			tree['right']=pruning_CART(tree['right'],right_data)

	if not isTree(tree['left']) and not isTree(tree['right']):
		left_data,right_data=data_split(test_data,tree['index'],tree['value'])
		if not left_data or not right_data:
			return getMean(tree)
		left_test_label=[d[-1] for d in left_data]
		right_test_label=[d[-1] for d in right_data]
		test_label=[d[-1] for d in test_data]
		####未剪枝前的错误率
		left_err_ratio=1-left_test_label.count(tree['left'])/len(left_test_label)
		right_err_ratio=1-right_test_label.count(tree['right'])/len(right_test_label)
		err=len(left_test_label)/len(test_label)*left_err_ratio+\
			len(right_test_label)/len(test_label)*right_err_ratio

		pruning_label=getMean(tree)
		pruning_err=1-test_label.count(pruning_label)/len(test_label)
		if pruning_err<=err:
			print('pruning')
			return pruning_label
		else:
			return tree
	return tree


def  get_label(tree):
	groups=tree['groups']
	label_list=[]
	for group in groups:
		for d in group:
			label_list.append(d[-1])
	label=max(set(label_list),key=label_list.count)
	return label

def data_split(tree,test_data):
	index=tree['index'];value=tree['value']

	left=[];right=[]
	for data in test_data:
		if data[index]<value:
			left.append(data)
		else:
			right.append(data)
	return left,right

def is_tree(obj):
	return isinstance(obj, dict)

def pruning_CART(tree,test_data):
	if not test_data:
		return get_label(tree)
	if is_tree(tree['left']) or is_tree(tree['right']):
		left_test_data,right_test_data=data_split(tree,test_data)

		if is_tree(tree['left']):
			tree['left']=\
			pruning_CART(tree['left'], left_test_data)
		if is_tree(tree['right']):
			tree['right']=\
			pruning_CART(tree['right'], right_test_data)

	if not is_tree(tree['left']) and not is_tree(tree['right']):
		left_test_data,right_test_data=data_split(tree,test_data)
		if len(left_test_data)==0 or len(right_test_data)==0:
			return get_label(tree)

		####剪枝前的情况
		left_test_label=[d[-1] for d in left_test_data]
		right_test_label=[d[-1] for d in right_test_data]
		test_label=[d[-1] for d in test_data]

		left_err_ratio=1-left_test_label.count(tree['left'])/len(left_test_label)
		right_err_ratio=1-right_test_label.count(tree['right'])/len(right_test_label)
		err=len(left_test_label)/len(test_label)*left_err_ratio+\
		len(right_test_label)/len(test_label)*right_err_ratio

		###剪枝之后
		pruning_label=get_label(tree)
		pruning_err=1-test_label.count(pruning_label)/len(test_label)

		if pruning_err<err:
			return pruning_label
		else:
			return tree
	return tree