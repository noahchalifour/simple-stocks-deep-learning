from datetime import datetime

params = {
    'symbols': [
        'BTC-USD', 
        'BCH-USD', 
        'ETH-USD', 
        'LTC-USD'
    ],
    'start_date': datetime(2000, 1, 1).date(),
    'end_date': datetime.today(),
    'chunk': 10,
    'num_hidden_layers': 3,
    'lstm_units': 512,
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'epochs': None,
    'batch_size': 64,
    'steps_per_checkpoint': 5,
    'test_size': 0.2,
    'max_data': None,
    'model_type': 'regression'
}
