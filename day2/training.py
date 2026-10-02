from pathlib import Path
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_absolute_percentage_error
from features import ROOT, PREDICTION_CYCLE, build_features, input_columns


def make_pipeline(model, columns, categorical=False):
    numeric = [c for c in columns if c != 'charging_policy']
    number_steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
    if not categorical:
        number_steps.append(('scale', StandardScaler()))
    category = 'passthrough' if categorical else OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    preprocess = ColumnTransformer([
        ('numeric', Pipeline(number_steps), numeric),
        ('category', category, ['charging_policy']),
    ], sparse_threshold=0)
    return Pipeline([('preprocess', preprocess), ('model', model)])


def train(name, make_model, categorical=False, tuned_params=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    features = build_features(args.data_dir)
    batch1 = features[(features['batch_id'] == 1) & features['cycle_life'].notna()].reset_index(drop=True)
    columns = input_columns(features)
    groups = batch1['charging_policy']
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, valid_index = next(split.split(batch1, groups=groups))
    train_data, valid_data = batch1.iloc[train_index], batch1.iloc[valid_index]
    X, y = train_data[columns], train_data['cycle_life']
    train_groups = train_data['charging_policy']
    pipeline = make_pipeline(make_model(columns), columns, categorical)
    fit_params = {'model__cat_features': [len(columns)-1]} if categorical else {}
    n_splits = min(5, train_groups.nunique())
    if n_splits < 2:
        raise ValueError('CV를 계산하려면 학습 구간에 충전 프로토콜이 2개 이상 필요합니다.')
    fold_records = []
    for fold, (fit_index, cv_index) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, train_groups), 1):
        estimator = clone(pipeline)
        estimator.fit(X.iloc[fit_index], y.iloc[fit_index], **fit_params)
        prediction = estimator.predict(X.iloc[cv_index])
        fold_records.append({'fold': fold, 'train_cells': len(fit_index), 'valid_cells': len(cv_index),
            'MAPE (%)': mean_absolute_percentage_error(y.iloc[cv_index], prediction)*100})
    folds = pd.DataFrame(fold_records)
    pipeline.fit(X, y, **fit_params)
    valid_prediction = pipeline.predict(valid_data[columns])
    valid_mape = mean_absolute_percentage_error(valid_data['cycle_life'], valid_prediction)*100
    metadata = {
        'model': name, 'report_version': 2, 'train_batch': 1, 'batch1_cells': len(batch1),
        'train_cells': len(train_data), 'valid_cells': len(valid_data),
        'target': 'cycle_life', 'prediction_cycle': PREDICTION_CYCLE,
        'summary_cycles': [1, PREDICTION_CYCLE], 'delta_cycles': [10, 100],
        'current_cycles': [10, 50, 100], 'feature_columns': columns,
        'train_cv_mape': float(folds['MAPE (%)'].mean()),
        'train_cv_mape_std': float(folds['MAPE (%)'].std(ddof=0)),
        'valid_holdout_mape': float(valid_mape), 'paper_target_mape': 9.1,
        'cv': f'GroupKFold({n_splits}) on Batch 1 training split',
        'holdout': 'GroupShuffleSplit(test_size=0.2), charging_policy',
        'train_cell_ids': train_data['cell_id'].astype(int).tolist(),
        'valid_cell_ids': valid_data['cell_id'].astype(int).tolist(),
        'train_protocols': sorted(train_groups.unique().tolist()),
        'valid_protocols': sorted(valid_data['charging_policy'].unique().tolist()),
        'hyperparameter_tuning': tuned_params is not None,
        'tuned_params': tuned_params or {}, 'random_state': 42,
    }
    args.models_dir.mkdir(parents=True, exist_ok=True)
    with (args.models_dir / f'{name}.pkl').open('wb') as file:
        pickle.dump({'pipeline': pipeline, 'metadata': metadata}, file)
    (args.models_dir / f'{name}.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    args.results_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.results_dir / f'{name}_cv.csv', index=False)
    print(f'{name}: Train {len(train_data)}개 / Valid {len(valid_data)}개')
    print(f"Train (Batch 1 CV): {metadata['train_cv_mape']:.3f}%")
    print(f'Valid (Batch 1 Hold-out): {valid_mape:.3f}%')
