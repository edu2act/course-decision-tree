import pandas as pd
# abalone=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',
#     header=None,prefix='V')
abalone=pd.read_csv('abalone.csv',header=None)
print(abalone.head())
# print(abalone.columns)

# print(abalone.V0.groupby(abalone.V0).count())
# print(abalone.head())

label=abalone.pop(8)
dummy=abalone.pop(0)
extra_fea=pd.get_dummies(dummy)
abalone=pd.concat([abalone,extra_fea],axis=1)
abalone=pd.concat([abalone,label],axis=1)

print(abalone.head())
# import CART_Classifier

# tree=CART_Classifier.fit(abalone.values[:300],  max_depth=6,minsize=10)
# print(tree)
# import treePlotter_cart
# treePlotter_cart.createPlot(tree)

