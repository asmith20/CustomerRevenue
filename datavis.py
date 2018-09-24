import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime

data = pd.read_csv('trainf.csv')
data["transactionRevenue"] = data["transactionRevenue"].astype('float')
badcols = [x for x in data.columns if data[x].nunique(dropna=False)==1 ]
data.drop(badcols,axis=1,inplace=True)

df = data.groupby("fullVisitorId")["transactionRevenue"].sum().reset_index()

plt.figure()
plt.scatter(range(df.shape[0]), np.sort(np.log1p(df["transactionRevenue"].values)))
plt.xlabel('index')
plt.ylabel('Revenue')


pbought = (data["transactionRevenue"]>0).sum()
pi = pbought/data.shape[0]
print("Percent of instances where a product was purchased ",pi)
pcbought = (df["transactionRevenue"]>0).sum()
pc = pcbought/df.shape[0]
print("Percent of customers who brought a product ",pc)

#Look at correlation for quantitative variables
rr= data.corr()
sns.heatmap(rr)

#Look into browsers
datab = data.groupby('browser')['transactionRevenue'].agg(['size', 'count', 'mean'])
datab = datab.sort_values(by="count", ascending=False)
datab['size'] = datab['size']/data.shape[0]

figb , axb = plt.subplots(2,1)
axb[0].set_title('Browser Types')
axb[0].bar(datab.index.values[0:10],datab['size'].head(10))
axb[0].set_xlabel('Browser')
axb[0].set_ylabel('Percent')

axb[1].bar(datab.index.values[0:10],datab['count'].head(10))
axb[1].set_xlabel('Browser')
axb[1].set_ylabel('Revenue')


#Look into device types
datadt = data.groupby('deviceCategory')['transactionRevenue'].agg(['size', 'count', 'mean'])
datadt = datadt.sort_values(by="count", ascending=False)
datadt['size'] = datadt['size']/data.shape[0]

figdt , axdt = plt.subplots(2,1)
axdt[0].set_title('Device Types')
axdt[0].bar(datadt.index.values,datadt['size'])
axdt[0].set_xlabel('Device Type')
axdt[0].set_ylabel('Percent')

axdt[1].bar(datadt.index.values,datadt['count'])
axdt[1].set_xlabel('Device Type')
axdt[1].set_ylabel('Revenue')

#Look into OS
dataos = data.groupby('operatingSystem')['transactionRevenue'].agg(['size', 'count', 'mean'])
dataos = dataos.sort_values(by="count", ascending=False)
dataos['size'] = dataos['size']/data.shape[0]

figos , axos = plt.subplots(2,1)
axos[0].set_title('Operating Systems')
axos[0].bar(dataos.index.values[0:10],dataos['size'].head(10))
axos[0].set_xlabel('OS')
axos[0].set_ylabel('Percent')

axos[1].bar(dataos.index.values[0:10],dataos['count'].head(10))
axos[1].set_xlabel('OS')
axos[1].set_ylabel('Revenue')


#Looking into date dependence
data['date'] = data['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
datedf = data.groupby('date')['transactionRevenue'].agg(['size','count'])

figd , axd = plt.subplots(2,1)
axd[0].set_title('Visits by Date')
axd[0].plot_date(datedf.index.values,datedf['size'],'-')
axd[0].set_xlabel('Date')
axd[0].set_ylabel('Visits')

axd[1].set_title('Revenue by Date')
axd[1].plot_date(datedf.index.values,datedf['count'],'-')
axd[1].set_xlabel('Date')
axd[1].set_ylabel('Revenue')

plt.show()