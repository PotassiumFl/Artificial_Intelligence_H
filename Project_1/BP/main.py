import argparse
import random

from classification import classification
from regression import regression


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--regression-csv", type=str, default=None)
    p.add_argument("--class-dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_seed = random.randint(1, int(1e9))
    if args.regression_csv:
        regression(args.regression_csv, random_state=run_seed)
    if args.class_dir:
        classification(args.class_dir, random_state=run_seed)
