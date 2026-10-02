from sklearn.ensemble import RandomForestRegressor
from training import train


if __name__ == '__main__':
    train('random_forest',
          lambda columns: RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
