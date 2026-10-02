from pathlib import Path
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from features import ROOT, build_features


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--batches', type=int, nargs='+', choices=[2, 3], default=[2])
    args = parser.parse_args()
    paths = sorted(args.models_dir.glob('*.pkl'))
    if not paths:
        raise FileNotFoundError(f'저장된 모델이 없습니다: {args.models_dir}')
    artifacts = []
    for path in paths:
        with path.open('rb') as file:
            artifact = pickle.load(file)
        if artifact['metadata'].get('report_version') != 2:
            raise ValueError(f'{path.name}: Hold-out 정보가 없습니다. 수정된 학습 파일로 다시 학습해주세요.')
        artifacts.append(artifact)
    best_name = min(artifacts, key=lambda a: a['metadata']['train_cv_mape'])['metadata']['model']
    features = build_features(args.data_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for artifact in artifacts:
        model, metadata = artifact['pipeline'], artifact['metadata']
        for batch in args.batches:
            sub = features[(features['batch_id'] == batch) & features['cycle_life'].notna()].copy()
            if sub.empty:
                continue
            y = sub['cycle_life'].to_numpy()
            prediction = model.predict(sub[metadata['feature_columns']])
            name = metadata['model']
            mape = mean_absolute_percentage_error(y, prediction)*100
            records.append({'model': name, 'batch_id': batch, 'cells': len(sub),
                'selected_by_batch1_cv': name == best_name,
                'Train MAPE (%)': metadata['train_cv_mape'],
                'Valid MAPE (%)': metadata['valid_holdout_mape'],
                'Test MAPE (%)': mape,
                'RMSE': np.sqrt(mean_squared_error(y, prediction)),
                'MAE': mean_absolute_error(y, prediction), 'R2': r2_score(y, prediction)})
            fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
            ax.scatter(y, prediction, alpha=0.8)
            low, high = min(y.min(), prediction.min()), max(y.max(), prediction.max())
            ax.plot([low, high], [low, high], '--', color='gray')
            ax.set_xlabel('Actual Cycle Life')
            ax.set_ylabel('Predicted Cycle Life')
            ax.set_title(f'{name} - Batch {batch}')
            fig.savefig(args.results_dir / f'{name}_batch{batch}.png', dpi=150)
            plt.close(fig)
    scores = pd.DataFrame(records)
    scores.to_csv(args.results_dir / 'evaluation.csv', index=False)
    (args.results_dir / 'selected_model.json').write_text(json.dumps(
        {'model': best_name, 'selection': 'Batch 1 training split group CV MAPE'}, indent=2)+'\n')
    print(f'\nBatch 1 CV로 선택한 모델: {best_name}')
    print(scores.round(3).to_string(index=False))


if __name__ == '__main__':
    evaluate()
