import argparse
import random

from classification import classification, test_classification_all
from regression import regression, test_regression_all


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--regression-dir", type=str, default=None)
    p.add_argument("--classification-dir", type=str, default=None)
    p.add_argument("--model-out", type=str, default="checkpoints/model.pt")
    p.add_argument("--model-in", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.model_in:
        if args.classification_dir:
            test_classification_all(args.classification_dir, args.model_in)
        elif args.regression_dir:
            test_regression_all(args.regression_dir, args.model_in)
    else:
        run_seed = random.randint(1, int(1e9))
        if args.regression_dir:
            regression(
                args.regression_dir, random_state=run_seed, model_out=args.model_out
            )
        if args.classification_dir:
            classification(
                args.classification_dir, random_state=run_seed, model_out=args.model_out
            )
