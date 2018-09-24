import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error


#clean up data, get rid of unwanted columns
data = pd.read_csv('trainf.csv')
data["transactionRevenue"] = data["transactionRevenue"].astype('float')
badcols = [x for x in data.columns if data[x].nunique(dropna=False)==1 ]
data.drop(badcols + ['campaignCode','adwordsClickInfo','adContent','subContinent', 'sessionId'],axis=1,inplace=True)

data["transactionRevenue"].fillna(0, inplace=True)

#label encode categorical variables
cat_vars = [x for x in data.columns if data[x].dtype == "object" and x != 'fullVisitorId']
for col in cat_vars:
    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(list(data[col].values.astype('str')))
    data[col] = label_enc.transform(list(data[col].values.astype('str')))

y = data['transactionRevenue'].values
X = data
X.drop(['transactionRevenue'],axis=1,inplace=True)

#train test split data
x_train, x_test, y_train, y_test = train_test_split(X,y)

trainid = x_train['fullVisitorId'].values
testid = x_test['fullVisitorId'].values
x_train.drop(['fullVisitorId'],axis=1,inplace=True)
x_test.drop(['fullVisitorId'],axis=1,inplace=True)

#xgboost fit
param = {'max_depth':50, 'silent':1,'objective':'reg:linear','eval_metric':'rmse','learning_rate':.1}
num_rounds = 100
dtrain = xgb.DMatrix(x_train,label=y_train)
dtest = xgb.DMatrix(x_test)
xg = xgb.train(param,dtrain,num_rounds)
y_pred = xg.predict(dtest)

rmse = mean_squared_error(y_test,y_pred)

print(rmse)










