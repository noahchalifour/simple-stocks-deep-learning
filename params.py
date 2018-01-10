from datetime import datetime

params = {
    'symbols': ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'ADA-USD'],
    'start_date': datetime(2000, 1, 1).date(),
    'end_date': datetime.today(),
    'chunk': 10,
    'num_hidden_layers': 3,
    'lstm_units': 100,
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'epochs': 1,
    'batch_size': 1,
    'test_size': 0.2,
    'max_data': 50,
    'model_type': 'regression'
}