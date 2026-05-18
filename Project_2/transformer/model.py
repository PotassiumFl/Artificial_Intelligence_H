import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

import config


class SinPosEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


class TransformerNER(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        pad_token_id: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos = SinPosEncoding(d_model, max_len, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.out = nn.Linear(d_model, num_tags)

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos(x)
        key_padding_mask = ~attention_mask
        return self.encoder(x, src_key_padding_mask=key_padding_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.encode(input_ids, attention_mask)
        return self.out(x)


def save_transformer_checkpoint(
    path: str | Path,
    model: TransformerNER,
    token2id: dict[str, int],
    tag2id: dict[str, int],
    id2tag: list[str],
    random_state: int,
    lang: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "token2id": token2id,
        "tag2id": tag2id,
        "id2tag": id2tag,
        "random_state": random_state,
        "lang": lang,
        "pad_token": config.PAD_TOKEN,
        "unk_token": config.UNK_TOKEN,
        "hparams": {
            "d_model": config.D_MODEL,
            "nhead": config.NHEAD,
            "num_encoder_layers": config.NUM_ENCODER_LAYERS,
            "dim_feedforward": config.DIM_FEEDFORWARD,
            "dropout": config.DROPOUT,
            "max_seq_len": config.MAX_SEQ_LEN,
        },
    }
    torch.save(payload, path)


def load_transformer_model(
    path: str | Path, device: torch.device
) -> tuple[TransformerNER, dict[str, Any]]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    hp = ckpt["hparams"]
    token2id: dict[str, int] = ckpt["token2id"]
    tag2id: dict[str, int] = ckpt["tag2id"]
    pad_id = int(token2id[ckpt.get("pad_token", config.PAD_TOKEN)])
    model = TransformerNER(
        vocab_size=len(token2id),
        num_tags=len(tag2id),
        pad_token_id=pad_id,
        d_model=int(hp["d_model"]),
        nhead=int(hp["nhead"]),
        num_layers=int(hp["num_encoder_layers"]),
        dim_feedforward=int(hp["dim_feedforward"]),
        dropout=float(hp["dropout"]),
        max_len=int(hp["max_seq_len"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt
