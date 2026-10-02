from importlib import import_module
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

common = import_module('00_data')
display = common.display


def main(context=None):
    context = context or common.get_context()
    df = context.df
    DATA_DIR = context.data_dir
    batch_ids, batch_names = context.batch_ids, context.batch_names
    cycle_life_df, life = context.cycle_life_df, context.life
    norm, cmap = context.norm, context.cmap

    print(f"DataFrame shape : {df.shape}")
    display(cycle_life_df.groupby('batch_id')['cycle_life'].agg(
        battery_count='size', valid_count='count', min='min',
        median='median', mean='mean', max='max',
    ).round(2))
    for batch_id in batch_ids:
        print(f"Batch {batch_id}: {batch_names[batch_id]}")

    # 히스토그램 / Box Plot
    fig, axes = plt.subplots(len(batch_ids), 2, figsize=(14, 4 * len(batch_ids)),
                             squeeze=False, layout='constrained')
    bins = np.linspace(150, 2300, 31)

    for row, batch_id in enumerate(batch_ids):
        values = cycle_life_df.loc[cycle_life_df['batch_id'] == batch_id, 'cycle_life'].dropna()
        ax = axes[row, 0]
        _, edges, bars = ax.hist(values, bins=bins, edgecolor='white')
        for bar, center in zip(bars, (edges[:-1] + edges[1:]) / 2):
            bar.set_facecolor(cmap(norm(center)))
        ax.axvline(500, color='tomato', linestyle='--')
        ax.axvline(1000, color='steelblue', linestyle='--')
        ax.set_xlim(150, 2300)
        ax.set_title(f'Batch {batch_id} - Histogram')
        ax.set_xlabel('Cycle Life')
        ax.set_ylabel('num of Battery Cell')

        ax = axes[row, 1]
        if len(values):
            ax.boxplot(values, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor='lightgray', alpha=0.6))
            ax.scatter(np.ones(len(values)), values, c=values, cmap=cmap, norm=norm,
                       s=18, alpha=0.7, zorder=3)
        ax.set_ylim(min(0, life.min()), life.max() * 1.05)
        ax.set_title(f'Batch {batch_id} - Box Plot')
        ax.set_ylabel('Cycle Life')
        ax.set_xticks([])
        print(f"Batch {batch_id}: <150 {(values < 150).sum()}개 / >2300 {(values > 2300).sum()}개")

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), label='Cycle Life')
    plt.show()

    # 장수명/단수명 비율
    records = []
    for batch_id in batch_ids:
        values = cycle_life_df.loc[cycle_life_df['batch_id'] == batch_id, 'cycle_life'].dropna()
        counts = [(values < 500).sum(), values.between(500, 1000).sum(), (values > 1000).sum()]
        for label, count in zip(['단수명 (<500)', '중간수명 (500~1000)', '장수명 (>1000)'], counts):
            records.append({'batch_id': batch_id, '수명 구분': label, '배터리 수': count,
                            '비율 (%)': count / len(values) * 100 if len(values) else np.nan})
    life_ratio_df = pd.DataFrame(records)
    display(life_ratio_df.round(2))

    # 단수명 셀 vs 그 외 셀 - Feature 분포
    short_life_df = df[df['cycle_life'] <= 500].copy()
    other_life_df = df[df['cycle_life'] > 500].copy()
    features = ['QD', 'QC', 'IR', 'Tmax', 'Tavg', 'Tmin', 'chargetime']

    fig, axes = plt.subplots(len(features), len(batch_ids),
                             figsize=(6 * len(batch_ids), 3 * len(features)),
                             squeeze=False, sharex='row', sharey='row', layout='constrained')
    for row, feature in enumerate(features):
        values = df.loc[df['cycle_life'].notna(), feature].replace([np.inf, -np.inf], np.nan).dropna()
        bins = np.histogram_bin_edges(values, bins=30)
        for col, batch_id in enumerate(batch_ids):
            ax = axes[row, col]
            for group, label, color in [(short_life_df, 'Life <= 500', 'tomato'),
                                         (other_life_df, 'Life > 500', 'steelblue')]:
                data = group.loc[group['batch_id'] == batch_id, feature]
                data = data.replace([np.inf, -np.inf], np.nan).dropna()
                if len(data):
                    ax.hist(data, bins=bins, density=True, color=color, alpha=0.5,
                            edgecolor='white', label=label)
            ax.set_title(f'Batch {batch_id} - {feature}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
    plt.show()

    # IR 구간별 단수명/그 외 비율
    ir_df = df.loc[df['cycle_life'].notna(), ['batch_id', 'IR', 'cycle_life']].copy()
    ir_df['IR'] = ir_df['IR'].replace([np.inf, -np.inf], np.nan)
    ir_df = ir_df.dropna(subset=['IR'])
    ir_df['수명 구분'] = np.where(ir_df['cycle_life'] <= 500, '단수명', '그 외')
    ir_edges = np.arange(14, 24) / 1000
    ir_bins = [-np.inf, *ir_edges, np.inf]
    ir_labels = ['0.014 미만'] + [
        f'{lower:.3f} 이상 ~ {upper:.3f} 미만'
        for lower, upper in zip(ir_edges[:-1], ir_edges[1:])
    ] + ['0.023 이상']
    ir_df['IR 구간 (Ω)'] = pd.cut(ir_df['IR'], bins=ir_bins, labels=ir_labels, right=False)

    ir_tables = []
    for batch_id in batch_ids:
        sub = ir_df[ir_df['batch_id'] == batch_id]
        table = pd.crosstab(sub['IR 구간 (Ω)'], sub['수명 구분'])
        table = table.reindex(index=ir_labels, columns=['단수명', '그 외'], fill_value=0)
        table.columns = ['단수명 측정값 수', '그 외 측정값 수']
        table['전체 측정값 수'] = table.sum(axis=1)
        for group in ['단수명', '그 외']:
            total = table[f'{group} 측정값 수'].sum()
            table[f'{group} 내 비율 (%)'] = table[f'{group} 측정값 수'] / total * 100 if total else np.nan
        print(f'Batch {batch_id}')
        display(table.round(2))
        ir_tables.append(table.assign(batch_id=batch_id).reset_index())
    ir_ratio_df = pd.concat(ir_tables, ignore_index=True)

    # 장수명 이상치 셀
    q1 = life.quantile(0.25)
    q3 = life.quantile(0.75)
    upper = q3 + 1.5 * (q3 - q1)
    outlier_cells = cycle_life_df[cycle_life_df['cycle_life'] > upper]
    print(f"전체 배치 공통 이상치 기준: {upper:.0f}사이클 초과")
    for batch_id in batch_ids:
        print(f'Batch {batch_id}')
        display(outlier_cells[outlier_cells['batch_id'] == batch_id].sort_values('cycle_life', ascending=False))

    # 초기 100사이클 평균 비교
    features = ['QD', 'QC', 'IR', 'Tmax', 'Tavg', 'Tmin', 'chargetime']
    early = df[df['cycle'].between(1, 100)].copy()
    early[features] = early[features].replace([np.inf, -np.inf], np.nan)
    cell_mean = early.groupby(['batch_id', 'cell_id'])[features].mean()
    cell_mean = cell_mean.join(cycle_life_df.set_index(['batch_id', 'cell_id'])['cycle_life'])
    cell_mean = cell_mean.dropna(subset=['cycle_life'])
    comparison_tables = []
    for batch_id in batch_ids:
        sub = cell_mean.xs(batch_id) if batch_id in cell_mean.index.get_level_values(0) else cell_mean.iloc[:0]
        long_mean = sub.loc[sub['cycle_life'] > upper, features].mean()
        other_mean = sub.loc[sub['cycle_life'] <= upper, features].mean()
        table = pd.DataFrame({'장수명 이상치 평균': long_mean, '그 외 평균': other_mean})
        table['차이 (%)'] = (long_mean - other_mean) / other_mean.replace(0, np.nan) * 100
        table.index.name = 'Feature'
        print(f'Batch {batch_id} - 초기 100사이클 평균')
        display(table.round(4))
        comparison_tables.append(table.assign(batch_id=batch_id).reset_index())
    comparison = pd.concat(comparison_tables, ignore_index=True)

if __name__ == '__main__':
    common.run_cli(main)
