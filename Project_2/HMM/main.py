import argparse

from ner import predict_hmm, train_hmm


def _parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--data-dir", type=str, required=True)
    train.add_argument("--model-out", type=str, required=True)

    predict = sub.add_parser("predict")
    predict.add_argument("--input", type=str, required=True)
    predict.add_argument("--output", type=str, required=True)
    predict.add_argument("--model-in", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    parsed = _parse_args()
    if parsed.cmd == "train":
        train_hmm(parsed.data_dir, parsed.model_out)
    elif parsed.cmd == "predict":
        predict_hmm(parsed.input, parsed.output, parsed.model_in)
    else:
        raise AssertionError
