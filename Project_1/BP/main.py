import argparse
import random

from classification import classification, test_classification_all
from regression import regression, test_regression_all


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--regression-csv", type=str, default=None)
    p.add_argument("--class-dir", type=str, default=None)
    p.add_argument("--model-out", type=str, default="checkpoints/model.pkl")
    p.add_argument("--model-in", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.model_in:
        if args.class_dir:
            test_classification_all(args.class_dir, args.model_in)
        elif args.regression_csv:
            test_regression_all(args.regression_csv, args.model_in)
    else:
        run_seed = random.randint(1, int(1e9))
        if args.regression_csv:
            regression(
                args.regression_csv, random_state=run_seed, model_out=args.model_out
            )
        if args.class_dir:
            classification(
                args.class_dir, random_state=run_seed, model_out=args.model_out
            )
