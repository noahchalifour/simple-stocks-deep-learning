from keras.models import model_from_json
from pandas_datareader import DataReader
from datetime import datetime
import os
import json
import numpy as np

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

model_name = '2018-01-07_0'

with open(os.path.join(FILE_DIR, 'models/{}/model.json'.format(model_name)), 'r') as f:
    model = model_from_json(f.read())

model.load_weights(os.path.join(FILE_DIR, 'models/{}/model.h5'.format(model_name)))

with open(os.path.join(FILE_DIR, 'models/{}/params.json'.format(model_name)), 'r') as f:
    params = json.load(f)



print('Loaded model {}.'.format(model_name))

stock_ticker = input("Enter a stock symbol: ")

# df = None
i = 0
# while df is None:
#     try:
if i == 5:
    raise Exception('Error getting data.')
i += 1
df = DataReader(stock_ticker.strip(), 'yahoo', start=params['start_date'], end=datetime.today())
    # except

x = np.array(list(df['Adj Close'][len(df['Adj Close'])-params['chunk']:]))

prediction = model.predict([x])

print(prediction)