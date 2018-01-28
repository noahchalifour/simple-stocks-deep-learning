from datetime import datetime
from pandas_datareader import DataReader
from pandas_datareader._utils import RemoteDataError
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import random
import math
import os
import pickle
import json
import time
import numpy as np

from params import params


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
read_data_timeout = 10


def get_data():

	data, all_closes = [], []

	for symbol in params['symbols']:
		df = None
		i = 0
		while df is None:
			try:
				if i == read_data_timeout:
					print('skipping', symbol)
					break
				i += 1
				df = DataReader(symbol, 'yahoo', start=params['start_date'], end=params['end_date'])
			except RemoteDataError:
				print('get_data_error', symbol)
				continue

		adj_close = df['Adj Close']
		all_closes += list(adj_close)

		for i in range(len(adj_close)):
			j = 0
			try:
				x, y = [], []
				for _ in range(params['chunk']):
					x.append(adj_close[i+j])
					j += 1
				y.append(adj_close[i+j])
				data.append((x, y))
			except IndexError:
				break

	return data, all_closes


def regression_model():

	data, all_closes = get_data()

	random.shuffle(data)

	scaler = MinMaxScaler(feature_range=(0, 1))
	all_closes = np.array(all_closes)
	all_closes = all_closes.reshape(-1, 1)
	scaler.fit(all_closes)

	if params['max_data'] is not None:
		if len(data) > params['max_data']:
			data = data[:params['max_data']]

	test_size = int(len(data) * params['test_size'])
	train_size = len(data) - test_size
	train_data = data[:train_size]
	test_data = data[train_size:]

	x_train = np.array([scaler.transform(np.array(x[0]).reshape(-1, 1)) for x in train_data])
	y_train = np.array([scaler.transform(np.array(x[1]).reshape(-1, 1))[0][0] for x in train_data])
	x_test = np.array([scaler.transform(np.array(x[0]).reshape(-1, 1)) for x in test_data])
	y_test = np.array([scaler.transform(np.array(x[1]).reshape(-1, 1))[0][0] for x in test_data])

	model = Sequential()
	model.add(LSTM(params['lstm_units'], input_shape=(params['chunk'], 1), return_sequences=True))

	for _ in range(params['num_hidden_layers']-1):
		model.add(LSTM(params['lstm_units'], return_sequences=True))

	model.add(LSTM(params['lstm_units']))
	model.add(Dense(1))
	model.compile(loss=params['loss'], optimizer=params['optimizer'])
	# print(model.summary())
	return model, scaler, x_train, y_train, x_test, y_test


if __name__ == '__main__':

	######################################

	model, scaler, x_train, y_train, x_test, y_test = regression_model()

	######################################

	print("Total data size: " + str(len(x_train) + len(x_test)))
	print("Train data size: " + str(len(x_train)))
	print("Test data size: " + str(len(x_test)))

	print("\nTraining...\n")

	today_date_str = str(datetime.today().date())
	same_date = [x for x in os.listdir(os.path.join(FILE_DIR, 'models')) if today_date_str in x]
	filename = '{}_{}'.format(today_date_str, len(same_date))

	if filename not in os.listdir(os.path.join(FILE_DIR, 'models')):
		os.makedirs(os.path.join(FILE_DIR, 'models/{}'.format(filename)))

	if params['epochs'] is not None:

		model.fit(x_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'])

		test_predictions = model.predict(x_test)
		y_test = scaler.inverse_transform([y_test])
		test_score = math.sqrt(mean_squared_error(y_test[0], test_predictions[:,0]))
		test_score_string = 'Test Score: %.2f RMSE' % test_score
		print(test_score_string)

		if not os.path.exists(os.path.join(FILE_DIR, 'models/{}/checkpoints/checkpoint-{}'.format(filename, params['epochs']))):
			os.makedirs(os.path.join(FILE_DIR, 'models/{}/checkpoints/checkpoint-{}'.format(filename, params['epochs'])))

		with open(os.path.join(FILE_DIR, 'models/{}/checkpoints/checkpoint-{}/model.json'.format(filename, params['epochs'])), 'w') as f:
			f.write(model.to_json())

		model.save_weights(os.path.join(FILE_DIR, 'models/{}/checkpoints/checkpoint-{}/model.h5'.format(filename, params['epochs'])))

		with open(os.path.join(FILE_DIR, 'models/{}/test_score.txt'.format(filename)), 'w') as f:
			f.write(test_score_string)

		with open(os.path.join(FILE_DIR, 'models/{}/scaler.pkl'.format(filename)), 'wb') as f:
			pickle.dump(scaler, f)

		params['start_date'] = str(params['start_date'])
		params['end_date'] = str(params['end_date'])
		with open(os.path.join(FILE_DIR, 'models/{}/params.json'.format(filename)), 'w') as f:
			json.dump(params, f)

		print("Model saved.")

	else:

		with open(os.path.join(FILE_DIR, 'models/{}/scaler.pkl'.format(filename)), 'wb') as f:
			pickle.dump(scaler, f)

		params['start_date'] = str(params['start_date'])
		params['end_date'] = str(params['end_date'])
		with open(os.path.join(FILE_DIR, 'models/{}/params.json'.format(filename)), 'w') as f:
			json.dump(params, f)

		start_time = time.time()
		prev_step = 0
		step = 0

		while True:
			model.fit(x_train, y_train, epochs=1, batch_size=params['batch_size'], verbose=0)

			step += len(x_train) // params['batch_size']

			for j in range(prev_step, step):
				if j % params['steps_per_checkpoint'] == 0 and j != 0:

					print("step: {}, time: {}".format(step, time.time() - start_time))
					if not os.path.exists(os.path.join(FILE_DIR, 'models/{}/checkpoints/checkpoint-{}'.format(filename, step))):
						os.makedirs(os.path.join(FILE_DIR, 'models/{}/checkpoints/checkpoint-{}'.format(filename, step)))
					with open(os.path.join(FILE_DIR, 'models/{}/checkpoints/checkpoint-{}/test_results.txt'.format(filename, step)), 'w') as f:
						for i in range(len(x_test)):
							prediction = model.predict(np.array([x_test[i]]))
							prediction = scaler.inverse_transform(prediction)[0][0]
							original_x = scaler.inverse_transform(x_test[i])
							f.write("#######################\n\nX: {}\nY Actual: {}\nY Predicted: {}\n\n".format(
								original_x[len(x_test[i])-1][0], 
								scaler.inverse_transform(y_test[i])[0][0],
								prediction
							))

					with open(os.path.join(FILE_DIR, 'models/{}/checkpoints/checkpoint-{}/model.json'.format(filename, step)), 'w') as f:
						f.write(model.to_json())

					model.save_weights(os.path.join(FILE_DIR, 'models/{}/checkpoints/checkpoint-{}/model.h5'.format(filename, step)))

			prev_step = step









