import argparse

from ner import predict_crf, train_crf


def _parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--data-dir", type=str, required=True)
    train.add_argument("--model-out", type=str, required=True)
    train.add_argument("--lang", type=str, required=True)

    predict = sub.add_parser("predict")
    predict.add_argument("--input", type=str, required=True)
    predict.add_argument("--output", type=str, required=True)
    predict.add_argument("--model-in", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    parsed = _parse_args()
    if parsed.cmd == "train":
        train_crf(parsed.data_dir, parsed.model_out, parsed.lang)
    elif parsed.cmd == "predict":
        predict_crf(parsed.input, parsed.output, parsed.model_in)
    else:
        raise AssertionError
