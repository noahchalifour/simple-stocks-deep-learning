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
    'chunk': 5,
    'num_hidden_layers': 3,
    'lstm_units': 100,
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'epochs': 3,
    'batch_size': 10,
    'steps_per_checkpoint': 5,
    'test_size': 0.2,
    'max_data': 50,
    'model_type': 'regression',
    'save_coreml': True
}
