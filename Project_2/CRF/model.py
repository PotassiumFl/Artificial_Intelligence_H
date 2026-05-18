from pathlib import Path
from typing import Any

import joblib
from sklearn_crfsuite import CRF

import config


class CRFTagger:
    def __init__(self, language: str):
        self.language = language.lower()
        self._crf = CRF(
            algorithm=config.CRF_ALGORITHM,
            c1=float(config.CRF_C1),
            c2=float(config.CRF_C2),
            max_iterations=int(config.CRF_MAX_ITERATIONS),
            all_possible_transitions=bool(config.ALL_POSSIBLE_TRANSITIONS),
        )

    def fit(self, X: list[list[dict[str, Any]]], tag_sequences: list[list[str]]) -> None:
        self._crf.fit(X, tag_sequences)

    def predict(self, X: list[list[dict[str, Any]]]) -> list[list[str]]:
        return self._crf.predict(X)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"crf": self._crf, "language": self.language}, p)

    @classmethod
    def load(cls, path: str | Path) -> "CRFTagger":
        blob = joblib.load(path)
        out = cls(language=str(blob["language"]))
        out._crf = blob["crf"]
        return out
