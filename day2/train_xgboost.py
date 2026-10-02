from xgboost import XGBRegressor
from training import train

PARAMS = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300, 'reg_lambda': 10}


if __name__ == '__main__':
    train('xgboost',
          lambda columns: XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=4, **PARAMS),
          tuned_params=PARAMS)
