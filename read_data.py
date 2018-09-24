import pandas as pd
import json
from pandas.io.json import json_normalize

def load_df(csv_path, nrows=None):
    json_col = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.read_csv(csv_path, dtype={'fullVisitorId': 'str'}, nrows=nrows)

    for column in json_col:
        df = df.join(pd.DataFrame(df.pop(column).apply(pd.io.json.loads).values.tolist(), index=df.index))
    return df

train = load_df('train.csv')
test = load_df('test.csv')

train.to_csv('trainf.csv',index=False)
test.to_csv('testf.csv',index=False)