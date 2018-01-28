from pandas_datareader import DataReader
from pandas_datareader._utils import RemoteDataError
from keras.models import model_from_json
from datetime import datetime
import os
import json
import pickle
import random
import argparse
import re
import numpy as np

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, type=str)
ap.add_argument('-c', '--checkpoint', required=True, type=str)
ap.add_argument('--start_date', required=True, type=str, help='Start test date in the form YYYY-MM-DD')
ap.add_argument('--end_date', required=True, type=str, help='End test date in the form YYYY-MM-DD')
ap.add_argument('-p', '--portfolio_value', required=False, type=int, default=10000.0, help='The starting investment amount (e.g: 10000)')
ap.add_argument('-s', '--steps_per_log', required=False, type=int, default=5, help='Number of days for every log')

args = vars(ap.parse_args())

start_date_sections = args['start_date'].split('-')
end_date_sections = args['end_date'].split('-')

portfolio = {
    'stocks': {},
    'value': args['portfolio_value']
}

test_start_date = datetime(int(start_date_sections[0]), int(start_date_sections[1]), int(start_date_sections[2]))
test_end_date = datetime(int(end_date_sections[0]), int(end_date_sections[1]), int(end_date_sections[2]))
model_name = args['model']
read_data_timeout = 5
days_per_log = args['steps_per_log']

compare_symbol = None


def sell_stock(symbol, num, date_index, portfolio, stock_data):

    portfolio['stocks'][symbol] -= num
    portfolio['value'] += (stock_data[date_index] * num)


def buy_stock(symbol, num, date_index, portfolio, stock_data):

    portfolio['stocks'][symbol] += num
    portfolio['value'] -= (stock_data[date_index] * num)


def get_portfolio_value(portfolio, date_index, all_data):

    value = portfolio['value']
    for stock, num in portfolio['stocks'].items():
        if all_data[stock] is None:
            continue
        value += (all_data[stock]['Adj Close'][date_index] * num)

    return value


if __name__ == '__main__':

    with open(os.path.join(FILE_DIR, 'models/{}/checkpoints/checkpoint-{}/model.json'.format(model_name, args['checkpoint'])), 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(os.path.join(FILE_DIR, 'models/{}/checkpoints/checkpoint-{}/model.h5'.format(model_name, args['checkpoint'])))

    with open(os.path.join(FILE_DIR, 'models/{}/scaler.pkl'.format(model_name)), 'rb') as f:
        scaler = pickle.load(f)

    with open(os.path.join(FILE_DIR, 'models/{}/params.json'.format(model_name)), 'r') as f:
        params = json.load(f)

    print('Loaded model {}.'.format(model_name))

    if compare_symbol is not None:
        if not isinstance(compare_symbol, str):
            raise Exception('compare_symbol must be a string.')

        compare_df = DataReader(compare_symbol, 'yahoo', start=test_start_date, end=test_end_date)
        compare_data = compare_df['Adj Close']

    all_data = {}
    for symbol in params['symbols']:
        all_data[symbol] = None
        i = 0
        while all_data[symbol] is None:
            try:
                if i == read_data_timeout:
                    print('skipping', symbol)
                    break
                i += 1
                all_data[symbol] = DataReader(symbol, 'yahoo', start=test_start_date, end=test_end_date)
            except RemoteDataError:
                print('get_data_error', symbol)
                continue

    # Init portfolio
    start_portfolio_value = portfolio['value']
    for symbol, data in all_data.items():
        portfolio['stocks'][symbol] = 0

    all_stocks = [k for (k, v) in all_data.items() if v is not None]

    j = 0
    for i in range(params['chunk'], len(all_data[params['symbols'][0]]['Adj Close'])):

        day_predictions = {}
        for symbol in all_data:
            if all_data[symbol] is None:
                continue

            x = np.array(all_data[symbol]['Adj Close'][j:i]).reshape(-1, 1)
            x = scaler.transform(x)
            x = np.array([x])

            prediction = model.predict(x)
            day_predictions[symbol] = scaler.inverse_transform(prediction)[0][0]

        if params['model_type'] == 'regression':

            positive_predictions = [(x, day_predictions[x]) for x in day_predictions if
                                    day_predictions[x] > all_data[x]['Adj Close'][i]]
            negative_predictions = [(x, day_predictions[x]) for x in day_predictions if
                                    day_predictions[x] < all_data[x]['Adj Close'][i]]

            # Short on negative stocks
            for stock, num in portfolio['stocks'].items():
                for negative_stock in negative_predictions:
                    if stock == negative_stock[0]:
                        sell_stock(symbol, num, i, portfolio, all_data[symbol]['Adj Close'])

            # Long on random positive choice
            positive_stock_choice = random.choice(positive_predictions)
            num_buy = portfolio['value'] // all_data[positive_stock_choice[0]]['Adj Close'][i]
            buy_stock(positive_stock_choice[0], num_buy, i, portfolio, all_data[positive_stock_choice[0]]['Adj Close'])

            invested_stocks = [k for (k, v) in portfolio['stocks'].items() if v > 0]

            if i % days_per_log == 0:
                print('date: {}, portfolio value: {}, invested stocks: {}'.format(
                    all_data[random.choice(all_stocks)].index[i],
                    round(get_portfolio_value(portfolio, i, all_data), 2),
                    invested_stocks
                ))

        j += 1

    end_portfolio_value = get_portfolio_value(portfolio, len(all_data[random.choice(all_stocks)]['Adj Close'])-1,
                                              all_data)

    print('\n######################################################')
    print('\nStart portfolio value:', start_portfolio_value)
    print('End portfolio value:', round(end_portfolio_value, 2))
    print('Change:', round((end_portfolio_value * 100) / start_portfolio_value, 2), '%\n')
    print('######################################################')