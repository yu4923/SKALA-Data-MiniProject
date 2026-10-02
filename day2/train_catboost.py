from catboost import CatBoostRegressor
from training import train


if __name__ == '__main__':
    train('catboost',
          lambda columns: CatBoostRegressor(
              loss_function='RMSE', boosting_type='Ordered', iterations=500,
              depth=6, learning_rate=0.03, l2_leaf_reg=3, random_seed=42,
              verbose=False, allow_writing_files=False, thread_count=4),
          categorical=True)
