from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support

import config
from data import load_sentences, read_corpus, write_predictions
from features import sentence_to_features
from model import CRFTagger


def _flatten_tag_sequences(seqs):
    return [tag for sentence in seqs for tag in sentence]


def _print_validation_report(
    tagger: CRFTagger, val_pairs: list[tuple[list[str], list[str]]]
) -> None:
    val_tokens = [p[0] for p in val_pairs]
    val_tags = [p[1] for p in val_pairs]
    val_features = [
        sentence_to_features(sentence, tagger.language) for sentence in val_tokens
    ]
    predicted_sequences = tagger.predict(val_features)
    flat_true = _flatten_tag_sequences(val_tags)
    flat_pred = _flatten_tag_sequences(predicted_sequences)
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


def train_crf(data_dir: str, model_out: str, lang: str) -> None:
    data_root = Path(data_dir)
    train_path = data_root / config.TRAIN_FILE
    train_pairs = read_corpus(train_path)

    tagger = CRFTagger(language=lang)
    train_tokens = [p[0] for p in train_pairs]
    train_tags = [p[1] for p in train_pairs]
    feature_sequences = [
        sentence_to_features(sentence, lang) for sentence in train_tokens
    ]
    tagger.fit(feature_sequences, train_tags)

    num_tag_types = len({t for tag_seq in train_tags for t in tag_seq})

    print("CRF 监督训练完成：" f"句子={len(train_pairs)} " f"tag 类数={num_tag_types}")
    model_path = Path(model_out)
    tagger.save(model_path)
    print(f"模型已保存: {model_path.resolve()}")

    val_path = data_root / config.VAL_FILE
    val_pairs = read_corpus(val_path)
    _print_validation_report(tagger, val_pairs)


def predict_crf(input_path: str, output_path: str, model_in: str) -> None:
    sentences = load_sentences(input_path)
    tagger = CRFTagger.load(model_in)
    feature_sequences = [
        sentence_to_features(sentence, tagger.language) for sentence in sentences
    ]
    predicted_sequences = tagger.predict(feature_sequences)
    write_predictions(output_path, sentences, predicted_sequences)
    print(f"已写入预测: {Path(output_path).resolve()} 句子数={len(sentences)}")
