from sklearn.ensemble import RandomForestRegressor
from training import train

PARAMS = {'max_depth': None, 'max_features': 0.7, 'min_samples_leaf': 1, 'n_estimators': 100}


if __name__ == '__main__':
    train('random_forest',
          lambda columns: RandomForestRegressor(random_state=42, n_jobs=-1, **PARAMS),
          tuned_params=PARAMS)
