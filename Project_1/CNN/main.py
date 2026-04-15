import argparse
import random

from classification import test_classifier_all, train_classifier


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--classification-dir", type=str, required=True)
    p.add_argument("--model-out", type=str, default="checkpoints/model.pt")
    p.add_argument("--model-in", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.model_in:
        test_classifier_all(args.classification_dir, args.model_in)
    else:
        run_seed = random.randint(1, int(1e9))
        train_classifier(
            args.classification_dir, random_state=run_seed, model_out=args.model_out
        )
