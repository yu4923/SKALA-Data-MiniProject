from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace
import argparse
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DATA_DIR = Path(os.environ.get('ESS_DATA_DIR', Path(__file__).resolve().parent.parent / 'data')).expanduser().resolve()
MAT_NAMES = [
    '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
]


def display(value):
    print(value.to_string() if hasattr(value, 'to_string') else value)
    print()


@lru_cache(maxsize=None)
def load_data(data_dir=DATA_DIR):
    data_dir = Path(data_dir).expanduser().resolve()
    mat_files = [data_dir / name for name in MAT_NAMES]
    # Summary 데이터 추출
    summary_columns = {
        "QDischarge": "QD", "QCharge": "QC", "IR": "IR",
        "Tmax": "Tmax", "Tavg": "Tavg", "Tmin": "Tmin",
        "chargetime": "chargetime",
    }
    frames = []
    for batch_id, path in enumerate(mat_files, start=1):
        with h5py.File(path, "r") as mat_file:
            batch_group = mat_file["batch"]
            summary_refs = batch_group["summary"][()].reshape(-1)
            life_refs = batch_group["cycle_life"][()].reshape(-1)
            policy_refs = batch_group["policy_readable"][()].reshape(-1)
            batch_rows = 0
            for local_cell_id, summary_ref in enumerate(summary_refs):
                summary = mat_file[summary_ref]
                values = {
                    column: np.asarray(summary[field][()]).reshape(-1)
                    for field, column in summary_columns.items()
                }
                n_cycles = len(values["QD"])
                if any(len(v) != n_cycles for v in values.values()):
                    raise ValueError(f"{path.name}, 셀 {local_cell_id}: summary 길이가 다릅니다.")
                cycle_numbers = (
                    np.asarray(summary["cycle"][()]).reshape(-1)
                    if "cycle" in summary else np.arange(1, n_cycles + 1)
                )
                cycle_life = float(np.asarray(mat_file[life_refs[local_cell_id]][()]).item())
                policy = "".join(
                    chr(int(v)) for v in mat_file[policy_refs[local_cell_id]][()].reshape(-1)
                )
                cell_df = pd.DataFrame(values)
                cell_df.insert(0, "charging_policy", policy)
                cell_df.insert(0, "cycle_life", cycle_life)
                cell_df.insert(0, "cycle", cycle_numbers)
                cell_df.insert(0, "cell_id", local_cell_id)
                cell_df["batch_id"] = batch_id
                cell_df["source_file"] = path.name
                frames.append(cell_df)
                batch_rows += len(cell_df)
            print(f"{path.name}: {len(summary_refs)}개 셀, {batch_rows:,}행")

    df = pd.concat(frames, ignore_index=True)

    return df


@lru_cache(maxsize=None)
def get_context(data_dir=DATA_DIR):
    df = load_data(data_dir)
    batch_ids = sorted(df['batch_id'].unique())
    batch_names = df.groupby('batch_id')['source_file'].first().to_dict()
    cycle_life_df = df.drop_duplicates(['batch_id', 'cell_id'])[
        ['batch_id', 'source_file', 'cell_id', 'cycle_life', 'charging_policy']
    ].copy()
    life = cycle_life_df['cycle_life'].dropna()
    norm = plt.Normalize(vmin=life.min(), vmax=life.max())
    cmap = plt.get_cmap('coolwarm_r')


    return SimpleNamespace(df=df, data_dir=Path(data_dir).expanduser().resolve(),
                           batch_ids=batch_ids, batch_names=batch_names,
                           cycle_life_df=cycle_life_df, life=life, norm=norm, cmap=cmap)


def run_cli(main):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--save-dir', type=Path)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    original_show = plt.show
    figure_count = 0

    def show(*show_args, **show_kwargs):
        nonlocal figure_count
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            for number in plt.get_fignums():
                figure_count += 1
                plt.figure(number).savefig(args.save_dir / f'{main.__module__}_{figure_count:02d}.png', dpi=150)
        if args.no_show:
            plt.close('all')
        else:
            original_show(*show_args, **show_kwargs)

    plt.show = show
    try:
        main(get_context(args.data_dir))
    finally:
        plt.show = original_show


if __name__ == '__main__':
    def main(context):
        print(f'통합 df: {context.df.shape}')
        display(context.df.head())
    run_cli(main)
