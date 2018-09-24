import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error


#clean up data, get rid of unwanted columns
data = pd.read_csv('trainf.csv')
testdata = pd.read_csv('testf.csv')
data["transactionRevenue"] = data["transactionRevenue"].astype('float')
badcols = [x for x in data.columns if data[x].nunique(dropna=False)==1 ]
data.drop(badcols + ['campaignCode','adwordsClickInfo','adContent','subContinent', 'sessionId'],axis=1,inplace=True)
testdata.drop(badcols + ['adwordsClickInfo','adContent','subContinent', 'sessionId'],axis=1,inplace=True)

data["transactionRevenue"].fillna(0, inplace=True)

#label encode categorical variables
cat_vars = [x for x in data.columns if data[x].dtype == "object" and x != 'fullVisitorId']
for col in cat_vars:
    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(list(data[col].values.astype('str')) + list(testdata[col].values.astype('str')))
    data[col] = label_enc.transform(list(data[col].values.astype('str')))
    testdata[col] = label_enc.transform(list(testdata[col].values.astype('str')))

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
param = {'max_depth':10, 'eta':0.01, 'silent':1,'objective':'reg:linear','eval_metric':'rmse','learning_rate':.01}
num_rounds = 1000
dtrain = xgb.DMatrix(x_train,label=y_train)
dtest = xgb.DMatrix(x_test)
xg = xgb.train(param,dtrain,num_rounds)
y_pred = xg.predict(dtest)

rmse = mean_squared_error(y_test,y_pred)


#compute predictions for actual test data
testid = testdata['fullVisitorId'].values
testdata.drop(['fullVisitorId'],axis=1,inplace=True)
out = pd.DataFrame({"fullVisitorId":testid})

dtest = xgb.DMatrix(testdata)
test_pred = xg.predict(dtest)
test_pred[test_pred < 0] = 0

out["PredictedLogRevenue"] = np.expm1(test_pred)
out = out.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
out.columns = ["fullVisitorId", "PredictedLogRevenue"]
out["PredictedLogRevenue"] = np.log1p(out["PredictedLogRevenue"])
out.to_csv("predictions.csv", index=False)