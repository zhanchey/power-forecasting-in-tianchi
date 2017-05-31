import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
import time
from sklearn import preprocessing

#preprocess
merged_data = pd.read_csv(r"E:\tianchi\Tianchi_power.csv",header=None, sep=',')
merged_data.columns = ['date', 'id', 'amount']
merged_data['date'] = pd.to_datetime(merged_data['date'], format='%Y/%m/%d')
#print(merged_data)
day_amount =merged_data.groupby('date').sum()
#print(day_amount)
day_amount.to_csv(r"E:\tianchi\day_amount.csv",index=True,sep=',')

#visual
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot_date(day_amount.index, day_amount.amount,\
            'b-',tz = None, xdate = True, ydate = False)
ax.set_title('Electric Power Load for Yangzhong')
ax.set_ylabel('Electric Load, MW')
plt.show()

day_data = pd.read_csv(r"E:\tianchi\day_amount_append.csv",header=None, sep=',')
day_data.columns = ['Date','Load','H_Temperature','L_Temperature']
#time in string format
data_df =pd.DataFrame(day_data['Date'],index = day_data.index)
#print((data_df))

#print(day_data)
#Create temporal predictors
day_data['Date'] = pd.to_datetime(day_data['Date'], format='%Y/%m/%d')
day_data['Year'] = pd.Series(
                    [day_data.Date[idx].year for idx in day_data.index],
                    index = day_data.index)
day_data['Month'] = pd.Series(
                    [day_data.Date[idx].month for idx in day_data.index],
                    index = day_data.index)
#There are important differences in load between weekdays and weekends
#Create suitable predictors:
day_data['DayOfWeek'] = pd.Series(
    [day_data.Date[idx].isoweekday() for idx in day_data.index],
    index = day_data.index)
day_data['isWeekend'] = pd.Series(
 [int(day_data.Date[idx].isoweekday() in [6,7]) for idx in day_data.index],
 index = day_data.index)
#There are important differences in load between holidays and non_holidays
#Create suitable predictors:
isholiday_list =[0]
isspringfestival_list =[0]
day_data['isholiday'] =pd.Series(isholiday_list,index = day_data.index)
day_data['isspringfestival'] =pd.Series(isspringfestival_list,index = day_data.index)
#print(day_data['isholiday'][0]+1)
for k in range(data_df.shape[0]):
    if(data_df['Date'][k] in ['2015/1/1','2015/1/2','2015/1/3','2015/3/5','2015/3/6','2015/4/5','2015/4/6','2015/5/1','2015/5/2','2015/5/3','2015/6/20','2015/6/21','2015/6/22','2015/9/3','2015/9/26','2015/9/27','2015/9/28','2015/10/1','2015/10/2','2015/10/3','2015/10/4','2015/10/5','2015/10/6','2015/10/7','2016/1/1','2016/1/2','2016/1/3','2016/4/3','2016/4/4','2016/4/5','2016/5/1','2016/5/2','2016/5/3']):
        day_data['isholiday'][k] =1
    if(data_df['Date'][k] in ['2015/2/10','2015/2/11','2015/2/12','2015/2/13','2015/2/14','2015/2/15','2015/2/16','2015/2/17','2015/2/18','2015/2/19','2015/2/20','2015/2/21','2015/2/22','2015/2/23','2015/2/24','2015/2/25','2015/2/26','2015/2/27','2015/2/28','2015/3/1','2016/2/3','2016/2/4','2016/2/5','2016/2/6','2016/2/7','2016/2/8','2016/2/9','2016/2/10','2016/2/11','2016/2/12','2016/2/13','2016/2/14','2016/2/15','2016/2/16','2016/2/17']):
        day_data['isspringfestival'][k] = 1

# day_data.to_csv(r"E:\tianchi\day_data.csv",index=True,sep=',')
#print(day_data)
#Lagged predictors:
day_data['PriorDay'] = day_data.Load.shift(1)
day_data['PriorDay'][0] =4020591
#print(day_data['PriorDay'])
day_data['Prior2Day'] = day_data.Load.shift(2)
day_data['Prior2Day'][0] =4170536;day_data['Prior2Day'][1] =4020591
#print(day_data['Prior2Day'])
day_data['PriorWeek']= day_data.Load.shift(7)
day_data['PriorWeek'][0] =4294518;day_data['PriorWeek'][1] =3874950;day_data['PriorWeek'][2] =3905750;day_data['PriorWeek'][3] =4105261;day_data['PriorWeek'][4] =4255276;day_data['PriorWeek'][5] =4170536;day_data['PriorWeek'][6] =4020591;
#print(day_data['PriorWeek'])
day_data = day_data.dropna()
#
features =['H_Temperature','L_Temperature','Month','DayOfWeek','isWeekend','isholiday','isspringfestival','PriorDay','Prior2Day','PriorWeek']
all =['H_Temperature','L_Temperature','Month','DayOfWeek','isWeekend','isholiday','isspringfestival','PriorDay','Prior2Day','PriorWeek','Load']
X = day_data[features]
Y = day_data.Load

#unnormalized version
# result =day_data[all]
# result.to_csv(r"E:\tianchi\predict\data.csv",index=True,sep=',')
# gradBoost = GradientBoostingRegressor(learning_rate = .1,
#                             n_estimators = 400, max_depth = 3)
# gradBoost.fit(X, Y)
# joblib.dump(gradBoost, "train_model.m")

#normalized version
result_normalized =day_data[all]
result_scaler = preprocessing.MinMaxScaler()
result_minmax = result_scaler.fit_transform(result_normalized)
#print((result.icol(2)))
#result_minmax.to_csv(r"E:\tianchi\predict\data_normalized.csv",index=True,sep=',')
np.savetxt(r"E:\tianchi\predict\data_normalized.csv", result_minmax, delimiter = ',')
gradBoost = GradientBoostingRegressor(learning_rate = .1,
                             n_estimators = 400, max_depth = 3)
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)
gradBoost.fit(X_minmax, Y)
joblib.dump(gradBoost, "normalized_train_model.m")
joblib.dump(min_max_scaler, "normalize_scaler.m")