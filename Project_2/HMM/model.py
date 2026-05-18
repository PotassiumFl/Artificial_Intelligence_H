from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


class HMMNER:
    def __init__(
        self,
        laplace_init: float = 1e-3,
        laplace_transition: float = 1e-3,
        laplace_emission: float = 1e-3,
        unk_min_freq: int = 1,
    ):
        self.laplace_pi = float(laplace_init)
        self.laplace_t = float(laplace_transition)
        self.laplace_e = float(laplace_emission)
        self.unk_min_freq = int(unk_min_freq)

        self.tags: list[str] = []
        self.word_to_ix: dict[str, int] = {}
        self.initial_log: np.ndarray | None = None
        self.trans_log: np.ndarray | None = None
        self.emit_log: np.ndarray | None = None

    def _build_vocab(self, sentences: list[list[str]]) -> None:
        freq: dict[str, int] = defaultdict(int)
        for s in sentences:
            for w in s:
                freq[w] += 1
        kept = sorted(w for w, c in freq.items() if c >= self.unk_min_freq)
        # 0 -> UNK/rare
        self.word_to_ix = {w: i + 1 for i, w in enumerate(kept)}

    def _word_ix(self, w: str) -> int:
        return int(self.word_to_ix.get(w, 0))

    def fit(
        self,
        sentence_pairs: list[tuple[list[str], list[str]]],
    ) -> None:
        pairs = sentence_pairs
        token_sents_f = [p[0] for p in pairs]
        tag_sents_f = [p[1] for p in pairs]

        self._build_vocab(token_sents_f)

        tag_vocab = sorted({t for tag_seq in tag_sents_f for t in tag_seq})
        self.tags = tag_vocab
        tag_to_ix = {t: i for i, t in enumerate(self.tags)}
        k = len(self.tags)
        v_sz = len(self.word_to_ix) + 1

        pi_counts = np.zeros(k, dtype=np.float64)
        a_counts = np.zeros((k, k), dtype=np.float64)
        b_counts = np.zeros((k, v_sz), dtype=np.float64)

        for toks, tag_seq in pairs:
            y0 = tag_to_ix[tag_seq[0]]
            pi_counts[y0] += 1
            for t, (w, tg) in enumerate(zip(toks, tag_seq)):
                yi = tag_to_ix[tg]
                b_counts[yi, self._word_ix(w)] += 1
            for t in range(1, len(toks)):
                a_counts[tag_to_ix[tag_seq[t - 1]], tag_to_ix[tag_seq[t]]] += 1

        eps = 1e-300
        lap_pi, lap_t, lap_e = self.laplace_pi, self.laplace_t, self.laplace_e

        pi = (pi_counts + lap_pi) / (pi_counts.sum() + lap_pi * k)
        pi = np.clip(pi, eps, None)
        pi /= pi.sum()

        a_rows = np.zeros((k, k), dtype=np.float64)
        for i in range(k):
            den = a_counts[i].sum() + lap_t * k
            if den > eps:
                a_rows[i] = (a_counts[i] + lap_t) / den
            else:
                a_rows[i] = np.ones(k) / k

        b_rows = np.zeros((k, v_sz), dtype=np.float64)
        for j in range(k):
            den = b_counts[j].sum() + lap_e * v_sz
            b_rows[j] = (b_counts[j] + lap_e) / den

        self.initial_log = np.log(np.clip(pi, eps, None))
        self.trans_log = np.log(np.clip(a_rows, eps, None))
        self.emit_log = np.log(np.clip(b_rows, eps, None))

    def predict_sentence(self, tokens: list[str]) -> list[str]:
        k = len(self.tags)
        n = len(tokens)
        if n == 0:
            return []
        wi = [self._word_ix(t) for t in tokens]
        score = np.zeros((n, k))
        back = np.zeros((n, k), dtype=np.int32)

        emit = self.emit_log
        pi = self.initial_log
        tr = self.trans_log
        score[0] = pi + emit[:, wi[0]]
        for t in range(1, n):
            prev = score[t - 1][:, np.newaxis] + tr

            idx_best = prev.argmax(axis=0)
            scr_best = prev.max(axis=0)
            score[t] = scr_best + emit[:, wi[t]]
            back[t] = idx_best.astype(np.int32)

        tags_idx = np.zeros(n, dtype=np.int32)
        tags_idx[n - 1] = int(score[n - 1].argmax())
        for t in range(n - 2, -1, -1):
            tags_idx[t] = int(back[t + 1, tags_idx[t + 1]])
        return [self.tags[int(ix)] for ix in tags_idx]

    def predict_corpus(self, sents: list[list[str]]) -> list[list[str]]:
        return [self.predict_sentence(s) for s in sents]

    def save(self, path: str | Path) -> None:

        payload = {
            "tags": self.tags,
            "words": sorted(self.word_to_ix.items(), key=lambda kv: kv[1]),
            "laplace_pi": self.laplace_pi,
            "laplace_t": self.laplace_t,
            "laplace_e": self.laplace_e,
            "unk_min_freq": self.unk_min_freq,
            "initial_log": (
                self.initial_log.tolist() if self.initial_log is not None else None
            ),
            "trans_log": (
                self.trans_log.tolist() if self.trans_log is not None else None
            ),
            "emit_log": self.emit_log.tolist() if self.emit_log is not None else None,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> HMMNER:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(
            laplace_init=raw["laplace_pi"],
            laplace_transition=raw["laplace_t"],
            laplace_emission=raw["laplace_e"],
            unk_min_freq=raw["unk_min_freq"],
        )
        m.tags = list(raw["tags"])
        m.word_to_ix = {w: int(i) for w, i in raw["words"]}
        if raw["initial_log"] is None:
            raise ValueError("模型文件损坏：缺少 initial_log")
        m.initial_log = np.array(raw["initial_log"], dtype=float)
        m.trans_log = np.array(raw["trans_log"], dtype=float)
        m.emit_log = np.array(raw["emit_log"], dtype=float)

        return m
