
import numpy as np
import matplotlib.pyplot as plt
y_test=[1,1,1,0,0,0,1,0,0,0]
# y_predprob=[0.49,0.94,0.56,0.05,0.53,0.27,0.9,0.58,0.32,0.21]
y_predprob=[0.75, 0.52,0.83,0.30, 0.13, 0.48,0.40,0.14,0.38, 0.30]
#因为要用到布尔值索引,所以将真实标签和预测的置信度分别转换为np.array类型list——>np.array
y_test=np.array(y_test)
y_predprob=np.array(y_predprob)

y_set=list(set(y_predprob))
##消除可能由于边界取不到的值
y_set=y_set+[-0.1,1.1]

y_set.sort(reverse=True)

###将y_set的每个值作为截断点

xy_arr=[]
for i in y_set:
	#先找到被预测为正类和负类的，真实的标签
    p1=y_test[y_predprob>=i]
    p0=y_test[y_predprob<i]
    #求tpr
    
    tpR=len(p1[p1==1])/(len(p1[p1==1])+len(p0[p0==1]))
    #求fpr
    fpR=len(p1[p1==0])/(len(p1[p1==0])+len(p0[p0==0]))
    
    xy_arr.append([fpR,tpR])

plt.plot([x[0] for x in xy_arr],[x[1] for x in xy_arr])
plt.plot([0,1],[0,1])
plt.show()
# #计算曲线下面积即AUC
# auc = 0.
# prev_x = 0
# for y,x in xy_arr:
#     if y != prev_x:
#         auc += (y - prev_x) * x
#         prev_x = y
# print( "the auc is %s."%auc)

# x = [_v[0] for _v in xy_arr]
# y = [_v[1] for _v in xy_arr]
# # plt.title("ROC curve of %s (AUC = %.4f)" % ('test' , 0.96))
# plt.ylabel("True Positive Rate")
# plt.xlabel("False Positive Rate")
# plt.plot(x ,y,c='y')
# # plt.plot([0,1],[1,0],c='g')
# plt.plot([0,1],[0,1],c='g')
# plt.show()