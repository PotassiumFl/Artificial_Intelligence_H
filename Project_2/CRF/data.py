from pathlib import Path

TaggedSentence = tuple[list[str], list[str]]


def iter_tokens_tags(path):
    path = Path(path)
    tokens: list[str] = []
    tags: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                if tokens:
                    yield tokens, tags
                    tokens, tags = [], []
                continue
            parts = raw.split()
            tokens.append(parts[0])
            tags.append(parts[1])
    if tokens:
        yield tokens, tags


def read_corpus(path) -> list[TaggedSentence]:
    out: list[TaggedSentence] = []
    for toks, tag_seq in iter_tokens_tags(path):
        if not toks:
            continue
        out.append((toks, tag_seq))
    return out


def load_sentences(path):
    path = Path(path)
    sentences: list[list[str]] = []
    current: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            tok = raw.split()[0]
            current.append(tok)
        if current:
            sentences.append(current)
    return sentences


def write_predictions(out_path, sentences_tokens, sentences_tags):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as w:
        first_sentence = True
        for tokens, preds in zip(sentences_tokens, sentences_tags):
            if not first_sentence:
                w.write("\n")
            first_sentence = False
            for tok, tag in zip(tokens, preds):
                w.write(f"{tok} {tag}\n")
