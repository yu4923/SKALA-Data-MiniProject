from pathlib import Path
import argparse
import runpy
import sys

MODELS = [
    'linear_regression', 'ridge', 'svr', 'random_forest', 'extra_trees',
    'gradient_boosting', 'catboost', 'xgboost', 'lightgbm',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--results-dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    options = []
    for name, value in vars(args).items():
        if value is not None:
            options.extend(['--' + name.replace('_', '-'), str(value)])
    original_argv = sys.argv[:]
    try:
        for name in MODELS:
            path = root / f'train_{name}.py'
            print(f'\n{name}', flush=True)
            sys.argv = [str(path), *options]
            runpy.run_path(str(path), run_name='__main__')
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
