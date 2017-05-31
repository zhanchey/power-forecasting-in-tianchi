import pandas as pd
import numpy as np
from sklearn.externals import joblib
#unnomalized
# model = joblib.load("train_model.m")
# data = pd.read_csv(r"E:\tianchi\predict\data_append.csv",header=None, sep=',')
# print(data)
# for i in range(610, 640):
#     data.iat[i, 8] =data.iat[i-1, 11]
#     data.iat[i, 9] =data.iat[i-2, 11]
#     data.iat[i, 10] =data.iat[i-7, 11]
#     a =model.predict(data.iloc[i,1:11])
#     a =int(a)
#     data.iat[i, 11] =a
#
# print(data)
# data.to_csv(r"E:\tianchi\predict\results.csv",index=True,sep=',')


#normalized
model = joblib.load("normalized_train_model.m")
scaler = joblib.load("normalize_scaler.m")
data = pd.read_csv(r"E:\tianchi\predict\data_normalized_append.csv",header=None, sep=',')
for i in range(609, 639):
    data.iat[i, 7] =data.iat[i-1, 10]
    data.iat[i, 8] =data.iat[i-2, 10]
    data.iat[i, 9] =data.iat[i-7, 10]
    s = list(data.ix[i, :])
    w = scaler.transform(s[0:10])
    print(w)
    a =model.predict(w)
    a =int(a)
    data.iat[i, 10] =a
#print(data)
data.to_csv(r"E:\tianchi\predict\results.csv",index=True,sep=',')