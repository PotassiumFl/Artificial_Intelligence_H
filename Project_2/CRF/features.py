def _slice(s: str, start: int, end: int) -> str:
    n = len(s)
    if start < 0:
        start = 0
    if end > n:
        end = n
    return s[start:end]


def features_zh(sentence: list[str], i: int) -> dict:
    w = sentence[i]
    feats: dict = {"bias": 1.0, "0": w, "len": str(len(w))}
    if len(w) >= 2:
        feats["ff2"] = w[:2]
        feats["lf2"] = w[-2:]
    if i > 0:
        p = sentence[i - 1]
        feats["-1"] = p
        feats["-1+0"] = p + "/" + w
    else:
        feats["BOS"] = True
    if i + 1 < len(sentence):
        n = sentence[i + 1]
        feats["+1"] = n
        feats["0+1"] = w + "/" + n
    else:
        feats["EOS"] = True
    if i > 1:
        feats["-2"] = sentence[i - 2]
    if i + 2 < len(sentence):
        feats["+2"] = sentence[i + 2]
    return feats


def features_en(sentence: list[str], i: int) -> dict:
    word = sentence[i]
    feats: dict = {
        "bias": 1.0,
        "wlower": word.lower(),
        "w[-3:]": _slice(word, max(0, len(word) - 3), len(word)),
        "w[-2:]": _slice(word, max(0, len(word) - 2), len(word)),
        "w[:3]": _slice(word, 0, min(3, len(word))),
        "w[:2]": _slice(word, 0, min(2, len(word))),
        "isupper": word.isupper(),
        "istitle": word.istitle(),
        "hasdigit": any(ch.isdigit() for ch in word),
        "hashyphen": "-" in word,
    }
    if i > 0:
        pw = sentence[i - 1]
        feats["-1:lower"] = pw.lower()
        feats["-1:w"] = pw
    else:
        feats["BOS"] = True
    if i + 1 < len(sentence):
        nw = sentence[i + 1]
        feats["+1:lower"] = nw.lower()
        feats["+1:w"] = nw
    else:
        feats["EOS"] = True
    if i > 1:
        feats["-2:lower"] = sentence[i - 2].lower()
    if i + 2 < len(sentence):
        feats["+2:lower"] = sentence[i + 2].lower()
    return feats


def sentence_to_features(sentence: list[str], language: str) -> list[dict]:
    lang = language.lower()
    if lang in ("zh", "cn", "chinese"):
        fn = features_zh
    elif lang in ("en", "english"):
        fn = features_en
    else:
        raise ValueError("language 请用 zh 或 en")
    return [fn(sentence, i) for i in range(len(sentence))]
