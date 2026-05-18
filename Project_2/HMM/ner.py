from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support

import config
from data import load_sentences, read_corpus, write_predictions
from model import HMMNER


def _flatten_tag_sequences(seqs):
    return [tag for sentence in seqs for tag in sentence]


def _print_validation_report(
    hmm: HMMNER, val_pairs: list[tuple[list[str], list[str]]]
) -> None:
    val_tokens = [p[0] for p in val_pairs]
    val_tags = [p[1] for p in val_pairs]
    predicted_tags = hmm.predict_corpus(val_tokens)
    flat_pred = _flatten_tag_sequences(predicted_tags)
    flat_true = _flatten_tag_sequences(val_tags)
    tag_set = sorted(set(flat_true) | set(flat_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_true,
        flat_pred,
        average="micro",
        labels=tag_set,
        zero_division=0,
    )
    print(
        "验证集 micro P/R/F1 = "
        f"{float(precision):.4f}/{float(recall):.4f}/{float(f1):.4f}"
    )


def train_hmm(data_dir: str, model_out: str) -> None:
    data_root = Path(data_dir)
    train_path = data_root / config.TRAIN_FILE
    train_pairs = read_corpus(train_path)

    hmm = HMMNER(
        laplace_init=config.LAPLACE_INIT,
        laplace_transition=config.LAPLACE_TRANSITION,
        laplace_emission=config.LAPLACE_EMISSION,
        unk_min_freq=config.UNK_MIN_FREQ,
    )

    model_path = Path(model_out)
    hmm.fit(train_pairs)
    num_tag_types = len(hmm.tags)
    print(
        "HMM 监督训练完成："
        f"句子={len(train_pairs)} "
        f"tag 类数={num_tag_types} "
        f"|词表|≈{len(hmm.word_to_ix)}"
    )
    hmm.save(model_path)
    print(f"模型已保存: {model_path.resolve()}")

    val_path = data_root / config.VAL_FILE
    val_pairs = read_corpus(val_path)
    _print_validation_report(hmm, val_pairs)


def predict_hmm(input_path: str, output_path: str, model_in: str) -> None:
    sentences = load_sentences(input_path)
    hmm = HMMNER.load(model_in)
    predicted_tags = hmm.predict_corpus(sentences)
    write_predictions(output_path, sentences, predicted_tags)
    print(f"已写入预测: {Path(output_path).resolve()} 句子数={len(sentences)}")
