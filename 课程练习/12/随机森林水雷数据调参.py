import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation, metrics

import matplotlib.pylab as plt
from sklearn.utils import shuffle
df=pd.read_csv('./data/sonar.all-data.csv',header=None,index_col=None)
#将字符改成类别标签
df[60]=df[60].map({'M':0,'R':1})
# print(df[60].value_counts())
labels=df.pop(60)

labels=labels.values
fea=df.values

X,y=shuffle(fea,labels,random_state=5)

rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(X,y)

print ('oob_score %f' % rf0.oob_score_)
y_predprob = rf0.predict_proba(X)[:,1]


print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))
#####第一种方法，利用交叉验证方法对变量逐一进行调整
# from sklearn.model_selection import cross_val_score

# def rmse_cv(model,X_train,y):
#     rmse = cross_val_score(model, X_train, y, 
#                                     scoring="roc_auc",
#                                    cv = 7)
#     return rmse

# cv_rf=rmse_cv(rf0,X,y)
# print(cv_rf)
# NS=range(50,101,10)
# cv_rf1=[rmse_cv(RandomForestClassifier(n_estimators=ns),X,y).mean() for ns in NS]
# print(cv_rf1)
# # dict(xgb_params, silent=1)
# cv_plt=pd.Series(cv_rf1,index=NS)
# plt.plot(cv_plt)
# plt.show()

#####第一种方法，利用交叉验证方法对变量逐一进行调整
#####第二种方法，网格搜索，每次调整多个变量
from sklearn.model_selection import GridSearchCV

# param_test1 = {'n_estimators':range(50,101,10)}
# gsearch1 = GridSearchCV(estimator = RandomForestClassifier(),
# 	param_grid = param_test1, scoring='roc_auc',cv=5)
# gsearch1.fit(X,y)
# print(gsearch1.grid_scores_)
# print(gsearch1.best_params_ )
# print(gsearch1.best_score_)
#####每次调整两个值
param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(2,11,2)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 70, 
                                  min_samples_leaf=2,max_features='auto' ),
   param_grid = param_test2, scoring='roc_auc', cv=5)
gsearch2.fit(X,y)
print(gsearch2.grid_scores_)
print(gsearch2.best_params_ )
print(gsearch2.best_score_)
#####每次调整两个值

rf1 = RandomForestClassifier(n_estimators= 70, max_depth=9, min_samples_split=2,
	min_samples_leaf=1,max_features=5 ,oob_score=True)
rf1.fit(X,y)
print ('oob_score %f' % rf1.oob_score_)

######接下来调整max_features
# param_test3 = {'max_features':range(3,11,2)}
# gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 70, max_depth=9, min_samples_split=2,
# 	min_samples_leaf=2 ,oob_score=True, random_state=10),
# param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
# gsearch3.fit(X,y)
# gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
# print(gsearch3.grid_scores_)
# print(gsearch3.best_params_ )
# print(gsearch3.best_score_)

######接下来调整max_features
param_test3 = {'min_samples_split':range(2,9,1),'min_samples_leaf':range(1,8,1)}
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 70, max_depth=9, oob_score=True, random_state=10),
	param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(X,y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
print(gsearch3.grid_scores_)
print(gsearch3.best_params_ )
print(gsearch3.best_score_)