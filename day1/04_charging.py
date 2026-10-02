from importlib import import_module
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

common = import_module('00_data')
display = common.display


def build_charging_features(context=None):
    context = context or common.get_context()
    df = context.df
    DATA_DIR = context.data_dir
    batch_ids, batch_names = context.batch_ids, context.batch_names
    cycle_life_df, life = context.cycle_life_df, context.life
    norm, cmap = context.norm, context.cmap

    # 충전 전류와 수명 비교
    current_records = []
    for batch_id in batch_ids:
        with h5py.File(DATA_DIR / batch_names[batch_id], 'r') as mat_file:
            batch = mat_file['batch']
            cycles_refs = batch['cycles'][()].reshape(-1)
            summary_refs = batch['summary'][()].reshape(-1)
            for _, cell in cycle_life_df[cycle_life_df['batch_id'] == batch_id].iterrows():
                cell_id = int(cell['cell_id'])
                summary = mat_file[summary_refs[cell_id]]
                cycle_numbers = summary['cycle'][()].reshape(-1)
                cycles = mat_file[cycles_refs[cell_id]]
                i_refs = cycles['I'][()].reshape(-1)
                t_refs = cycles['t'][()].reshape(-1)
                metrics = []
                for cycle_number in [10, 50, 100]:
                    indices = np.flatnonzero(cycle_numbers == cycle_number)
                    if not len(indices) or indices[0] >= min(len(i_refs), len(t_refs)):
                        continue
                    index = int(indices[0])
                    current = mat_file[i_refs[index]][()].reshape(-1)
                    time = mat_file[t_refs[index]][()].reshape(-1)
                    if len(current) != len(time):
                        raise ValueError(f'Batch {batch_id}, Cell {cell_id}: I와 t 길이가 다릅니다.')
                    dt = np.diff(time)
                    mid_current = (current[:-1] + current[1:]) / 2
                    valid = (dt > 0) & np.isfinite(dt) & np.isfinite(mid_current) & (mid_current > 0.1)
                    if not valid.any():
                        continue
                    weights = dt[valid]
                    values = mid_current[valid]
                    mean_current = np.average(values, weights=weights)
                    std_current = np.sqrt(np.average((values - mean_current) ** 2, weights=weights))
                    order = np.argsort(values)
                    cumulative = np.cumsum(weights[order]) / weights.sum()
                    p95 = values[order][np.searchsorted(cumulative, 0.95)]
                    metrics.append([mean_current, std_current, p95])
                if not metrics:
                    continue

                qd = summary['QDischarge'][()].reshape(-1)
                early = pd.DataFrame({'cycle': cycle_numbers, 'QD': qd})
                early = early.replace([np.inf, -np.inf], np.nan).dropna()
                early = early[early['cycle'].between(1, 100)].sort_values('cycle')
                rate = np.nan
                if len(early) >= 20 and early['cycle'].max() - early['cycle'].min() >= 50:
                    smooth_qd = early['QD'].rolling(21, center=True, min_periods=1).median()
                    rate = -np.polyfit(early['cycle'], smooth_qd, 1)[0] * 100
                mean_metrics = np.mean(metrics, axis=0)
                current_records.append({
                    'batch_id': batch_id, 'cell_id': cell_id,
                    'charging_policy': cell['charging_policy'], 'cycle_life': cell['cycle_life'],
                    'mean_current_A': mean_metrics[0], 'std_current_A': mean_metrics[1],
                    'p95_current_A': mean_metrics[2], 'early_QD_loss': rate,
                    'sample_cycles': len(metrics),
                })

    return pd.DataFrame(current_records)

def main(context=None):
    context = context or common.get_context()
    df = context.df
    DATA_DIR = context.data_dir
    batch_ids, batch_names = context.batch_ids, context.batch_names
    cycle_life_df, life = context.cycle_life_df, context.life
    norm, cmap = context.norm, context.cmap

    # 충전 프로토콜별 평균 수명
    policy_life = cycle_life_df.groupby(['batch_id', 'charging_policy'])['cycle_life'].agg(
        mean='mean', std='std', count='count',
    ).reset_index()
    max_life = (policy_life['mean'] + policy_life['std'].fillna(0)).max() * 1.15

    for batch_id in batch_ids:
        sub = policy_life[(policy_life['batch_id'] == batch_id) & (policy_life['count'] > 0)]
        sub = sub.sort_values('mean')
        fig, ax = plt.subplots(figsize=(12, max(5, len(sub) * 0.45)), layout='constrained')
        positions = np.arange(len(sub))
        ax.barh(positions, sub['mean'], xerr=sub['std'].fillna(0),
                color=cmap(norm(sub['mean'].to_numpy())), capsize=3, alpha=0.8)
        ax.set_yticks(positions)
        ax.set_yticklabels(sub['charging_policy'], fontsize=11)
        ax.set_xlim(0, max_life)
        ax.set_xlabel('Mean Cycle Life')
        ax.set_title(f'Batch {batch_id} - Charging Policy')
        for y, mean, std, count in zip(positions, sub['mean'], sub['std'].fillna(0), sub['count']):
            ax.text(mean + std + 20, y, f'{mean:.0f} (n={count})', va='center', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Cycle Life')
        plt.show()
        print(f'Batch {batch_id}')
        display(sub.sort_values('mean', ascending=False).round(2))


    charging_features = build_charging_features(context)
    fig, axes = plt.subplots(1, len(batch_ids), figsize=(7 * len(batch_ids), 5),
                             squeeze=False, sharex=True, sharey=True, layout='constrained')
    for ax, batch_id in zip(axes.flat, batch_ids):
        sub = charging_features[charging_features['batch_id'] == batch_id].dropna(subset=['cycle_life'])
        ax.scatter(sub['mean_current_A'], sub['cycle_life'],
                   c=sub['cycle_life'], cmap=cmap, norm=norm, alpha=0.8)
        ax.set_title(f'Batch {batch_id} - Charging Current vs Life')
        ax.set_xlabel('Mean Charging Current (A)')
        ax.set_ylabel('Cycle Life')
        ax.grid(alpha=0.3)

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), label='Cycle Life')
    plt.show()

    # 전체 구간 - 충전 전류 패턴과 열화 속도
    full_records = []
    for batch_id in batch_ids:
        print(f'Batch {batch_id} - 전체 사이클 계산 중...', flush=True)
        with h5py.File(DATA_DIR / batch_names[batch_id], 'r') as mat_file:
            batch = mat_file['batch']
            cycles_refs = batch['cycles'][()].reshape(-1)
            summary_refs = batch['summary'][()].reshape(-1)
            for _, cell in cycle_life_df[cycle_life_df['batch_id'] == batch_id].iterrows():
                cell_id = int(cell['cell_id'])
                cycles = mat_file[cycles_refs[cell_id]]
                i_refs = cycles['I'][()].reshape(-1)
                t_refs = cycles['t'][()].reshape(-1)
                total_time = 0.0
                sum_current = 0.0
                sum_current_squared = 0.0
                cycle_p95 = []

                for index in range(min(len(i_refs), len(t_refs))):
                    current = mat_file[i_refs[index]][()].reshape(-1)
                    time = mat_file[t_refs[index]][()].reshape(-1)
                    if len(current) != len(time):
                        raise ValueError(f'Batch {batch_id}, Cell {cell_id}: I와 t 길이가 다릅니다.')
                    dt = np.diff(time)
                    mid_current = (current[:-1] + current[1:]) / 2
                    valid = (dt > 0) & np.isfinite(dt) & np.isfinite(mid_current) & (mid_current > 0.1)
                    if not valid.any():
                        continue
                    weights = dt[valid]
                    values = mid_current[valid]
                    total_time += weights.sum()
                    sum_current += np.sum(values * weights)
                    sum_current_squared += np.sum(values ** 2 * weights)
                    order = np.argsort(values)
                    cumulative = np.cumsum(weights[order]) / weights.sum()
                    cycle_p95.append(values[order][np.searchsorted(cumulative, 0.95)])
                if total_time == 0:
                    continue

                summary = mat_file[summary_refs[cell_id]]
                qd = pd.DataFrame({
                    'cycle': summary['cycle'][()].reshape(-1),
                    'QD': summary['QDischarge'][()].reshape(-1),
                }).replace([np.inf, -np.inf], np.nan).dropna().sort_values('cycle')
                rate = np.nan
                if len(qd) >= 20 and qd['cycle'].nunique() > 1:
                    smooth_qd = qd['QD'].rolling(21, center=True, min_periods=1).median()
                    rate = -np.polyfit(qd['cycle'], smooth_qd, 1)[0] * 100

                mean_current = sum_current / total_time
                full_records.append({
                    'batch_id': batch_id, 'cell_id': cell_id, 'cycle_life': cell['cycle_life'],
                    'mean_current_A': mean_current,
                    'std_current_A': np.sqrt(max(0, sum_current_squared / total_time - mean_current ** 2)),
                    'p95_current_A': np.mean(cycle_p95), 'full_QD_loss': rate,
                    'current_cycles': len(cycle_p95), 'QD_cycles': len(qd),
                })

    full_charging_features = pd.DataFrame(full_records)
    metrics = ['mean_current_A', 'std_current_A', 'p95_current_A']
    correlation_records = []
    for batch_id in batch_ids:
        sub = full_charging_features[full_charging_features['batch_id'] == batch_id]
        for feature in metrics:
            for target in ['cycle_life', 'full_QD_loss']:
                pairs = sub[[feature, target]].dropna()
                correlation_records.append({
                    'batch_id': batch_id, '전류 Feature': feature, '비교 대상': target,
                    '셀 수': len(pairs),
                    'Pearson r': pairs[feature].corr(pairs[target]) if len(pairs) >= 3 else np.nan,
                    'Spearman r': pairs[feature].rank().corr(pairs[target].rank()) if len(pairs) >= 3 else np.nan,
                })
    full_current_correlation = pd.DataFrame(correlation_records)

    xmax = full_charging_features['mean_current_A'].max() * 1.1
    rates = full_charging_features['full_QD_loss'].dropna()
    ymin = min(0, rates.min())
    ymax = rates.max()
    padding = max((ymax - ymin) * 0.1, 0.0001)

    for batch_id in batch_ids:
        print(f'Batch {batch_id} - 전체 구간')
        display(full_current_correlation[full_current_correlation['batch_id'] == batch_id].round(3))
        sub = full_charging_features[full_charging_features['batch_id'] == batch_id].dropna(subset=['full_QD_loss'])
        colors = [cmap(norm(value)) if np.isfinite(value) else 'gray' for value in sub['cycle_life']]
        fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
        ax.scatter(sub['mean_current_A'], sub['full_QD_loss'], color=colors, s=70, alpha=0.8)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlim(0, xmax)
        ax.set_ylim(ymin - padding, ymax + padding)
        ax.set_title(f'Batch {batch_id} - Full Cycle Charging Current vs QD Loss')
        ax.set_xlabel('Mean Charging Current (A)')
        ax.set_ylabel('Full Cycle QD Loss (Ah / 100 cycles)')
        ax.grid(alpha=0.3)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Cycle Life')
        plt.show()


if __name__ == '__main__':
    common.run_cli(main)
