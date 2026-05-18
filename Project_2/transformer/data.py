from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

import config

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


def build_token_vocab(sentences, min_freq):
    freq = Counter()
    for sentence in sentences:
        for word in sentence:
            freq[word] += 1
    token2id = {config.PAD_TOKEN: 0, config.UNK_TOKEN: 1}
    for word, count in sorted(freq.items()):
        if count >= min_freq:
            token2id[word] = len(token2id)
    id2token = [""] * len(token2id)
    for word, index in token2id.items():
        id2token[index] = word
    return token2id, id2token


def build_tag_vocab(tag_sequences_per_sentence):
    all_tags = sorted({t for tag_seq in tag_sequences_per_sentence for t in tag_seq})
    tag2id = {t: index for index, t in enumerate(all_tags)}
    id2tag = all_tags
    return tag2id, id2tag


class NerDataset(Dataset):
    def __init__(self, sentences, tags, token2id, tag2id, max_len):
        self.sentences = sentences
        self.tags = tags
        self.token2id = token2id
        self.tag2id = tag2id
        self.max_len = max_len
        self._unk_index = token2id[config.UNK_TOKEN]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        tokens = self.sentences[index]
        length = min(len(tokens), self.max_len)
        input_ids = [
            self.token2id.get(tokens[i], self._unk_index) for i in range(length)
        ]
        if self.tags is None:
            return torch.tensor(input_ids, dtype=torch.long)
        tag_seq = self.tags[index]
        tag_ids = [self.tag2id[tag_seq[i]] for i in range(length)]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(
            tag_ids, dtype=torch.long
        )


def collate_train(batch):
    ids_list = [item[0] for item in batch]
    tags_list = [item[1] for item in batch]
    return _pad_batch(ids_list, tags_list)


def collate_infer(batch):
    return _pad_batch(batch, None)


def _pad_batch(ids_list, tags_list):
    pad_id = 0
    max_length = max(ids.numel() for ids in ids_list)
    batch_size = len(ids_list)
    input_ids = torch.full((batch_size, max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    tags_tensor = torch.full((batch_size, max_length), -100, dtype=torch.long)
    for row_index, ids in enumerate(ids_list):
        sequence_length = ids.numel()
        input_ids[row_index, :sequence_length] = ids
        attention_mask[row_index, :sequence_length] = True
        if tags_list is not None:
            tags_tensor[row_index, :sequence_length] = tags_list[row_index]
    return input_ids, attention_mask, tags_tensor
