import random
from pathlib import Path

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

import config
from data import (
    NerDataset,
    build_tag_vocab,
    build_token_vocab,
    collate_infer,
    collate_train,
    load_sentences,
    read_corpus,
    write_predictions,
)
from model import TransformerNER, load_transformer_model, save_transformer_checkpoint


def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _val_loss_improved(val_loss, best_val_loss, min_delta):
    if best_val_loss == float("inf"):
        return True
    return val_loss < (best_val_loss - min_delta)


def _micro_f1_on_pairs(model, val_pairs, token2id, tag2id, device):
    val_tokens = [p[0] for p in val_pairs]
    val_tags = [p[1] for p in val_pairs]
    dataset = NerDataset(val_tokens, val_tags, token2id, tag2id, config.MAX_SEQ_LEN)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_train,
    )
    model.eval()
    pred_flat: list[int] = []
    true_flat: list[int] = []
    with torch.no_grad():
        for input_ids, attention_mask, tags_batch in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask)
            pred_np = logits.argmax(dim=-1).cpu().numpy()
            gold_np = tags_batch.detach().cpu().numpy()
            attention_np = attention_mask.detach().cpu().numpy()
            for batch_index in range(pred_np.shape[0]):
                for time_index in range(pred_np.shape[1]):
                    if (
                        attention_np[batch_index, time_index]
                        and gold_np[batch_index, time_index] != -100
                    ):
                        pred_flat.append(int(pred_np[batch_index, time_index]))
                        true_flat.append(int(gold_np[batch_index, time_index]))
    if not true_flat:
        return 0.0, 0.0, 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_flat,
        pred_flat,
        average="micro",
        zero_division=0,
    )
    return float(precision), float(recall), float(f1)


def _print_validation_report(model, val_pairs, token2id, tag2id, device):
    precision, recall, f1 = _micro_f1_on_pairs(
        model, val_pairs, token2id, tag2id, device
    )
    print("验证集 micro P/R/F1 = " f"{precision:.4f}/{recall:.4f}/{f1:.4f}")


def train_transformer(data_dir: str, model_out: str, lang: str) -> None:
    random_state = config.TRAIN_RANDOM_SEED
    _set_seed(random_state)

    data_root = Path(data_dir)
    train_path = data_root / config.TRAIN_FILE
    val_path = data_root / config.VAL_FILE

    train_pairs = read_corpus(train_path)
    val_pairs = read_corpus(val_path)
    train_tokens = [p[0] for p in train_pairs]
    train_tags = [p[1] for p in train_pairs]
    val_tags = [p[1] for p in val_pairs]

    token2id, _ = build_token_vocab(train_tokens, config.UNK_MIN_FREQ)
    tag2id, id2tag = build_tag_vocab(train_tags + val_tags)

    num_tag_types = len(id2tag)
    print(
        "Transformer 监督训练："
        f"tag 类数={num_tag_types} "
        f"训练句={len(train_pairs)} "
        f"验证句={len(val_pairs)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerNER(
        vocab_size=len(token2id),
        num_tags=len(tag2id),
        pad_token_id=token2id[config.PAD_TOKEN],
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_ENCODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        max_len=config.MAX_SEQ_LEN,
    ).to(device)

    train_dataset = NerDataset(
        train_tokens, train_tags, token2id, tag2id, config.MAX_SEQ_LEN
    )
    val_tokens = [p[0] for p in val_pairs]
    val_dataset = NerDataset(val_tokens, val_tags, token2id, tag2id, config.MAX_SEQ_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_train,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_train,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    use_early_stop = config.EARLY_STOP_PATIENCE > 0 and len(val_dataset) > 0
    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for input_ids, attention_mask, tags_batch in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            tags_batch = tags_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tags_batch.view(-1))
            loss.backward()
            clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        val_pred_flat: list[int] = []
        val_true_flat: list[int] = []
        with torch.no_grad():
            for input_ids, attention_mask, tags_batch in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                tags_dev = tags_batch.to(device)
                logits = model(input_ids, attention_mask)
                val_loss_sum += criterion(
                    logits.view(-1, logits.size(-1)),
                    tags_dev.view(-1),
                ).item()
                val_batches += 1
                pred_np = logits.argmax(dim=-1).cpu().numpy()
                gold_np = tags_dev.detach().cpu().numpy()
                attention_np = attention_mask.detach().cpu().numpy()
                for batch_index in range(pred_np.shape[0]):
                    for time_index in range(pred_np.shape[1]):
                        if (
                            attention_np[batch_index, time_index]
                            and gold_np[batch_index, time_index] != -100
                        ):
                            val_pred_flat.append(int(pred_np[batch_index, time_index]))
                            val_true_flat.append(int(gold_np[batch_index, time_index]))

        val_loss = val_loss_sum / max(val_batches, 1)
        avg_train_loss = running_loss / max(n_batches, 1)
        if val_true_flat:
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_true_flat,
                val_pred_flat,
                average="micro",
                zero_division=0,
            )
        else:
            precision = recall = f1 = 0.0

        if use_early_stop:
            if _val_loss_improved(val_loss, best_val_loss, config.EARLY_STOP_MIN_DELTA):
                best_val_loss = val_loss
                best_state = {
                    tensor_name: tensor_value.detach().cpu().clone()
                    for tensor_name, tensor_value in model.state_dict().items()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= config.EARLY_STOP_PATIENCE:
                    print(
                        "早停于 epoch "
                        f"{epoch}/{config.EPOCHS}  best_val_loss={best_val_loss:.6f}"
                    )
                    break

        if (
            epoch == 1
            or epoch % max(1, config.EPOCHS // 10) == 0
            or epoch == config.EPOCHS
        ):
            print(
                f"Epoch {epoch:4d}/{config.EPOCHS}  "
                f"train_loss={avg_train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val micro P/R/F1="
                f"{float(precision):.4f}/{float(recall):.4f}/{float(f1):.4f}"
            )

    if use_early_stop and best_state is not None:
        loaded = {name: weights.to(device) for name, weights in best_state.items()}
        model.load_state_dict(loaded)
        model.eval()

    output_path = Path(model_out)
    save_transformer_checkpoint(
        output_path,
        model,
        token2id,
        tag2id,
        id2tag,
        random_state=random_state,
        lang=lang,
    )
    print(f"模型已保存: {output_path.resolve()}")

    _print_validation_report(model, val_pairs, token2id, tag2id, device)


@torch.no_grad()
def predict_transformer(input_path: str, output_path: str, model_in: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_transformer_model(model_in, device)
    token2id = checkpoint["token2id"]
    tag2id = checkpoint["tag2id"]
    id2tag = checkpoint["id2tag"]

    token_sents = load_sentences(input_path)
    infer_dataset = NerDataset(token_sents, None, token2id, tag2id, config.MAX_SEQ_LEN)
    loader = DataLoader(
        infer_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_infer,
    )
    model.eval()
    all_pred_indices: list[list[int]] = []
    for input_ids, attention_mask, _ in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        logits = model(input_ids, attention_mask)
        pred_np = logits.argmax(dim=-1).cpu().numpy()
        attention_np = attention_mask.detach().cpu().numpy()
        for batch_index in range(pred_np.shape[0]):
            sequence_ids: list[int] = []
            for time_index in range(pred_np.shape[1]):
                if attention_np[batch_index, time_index]:
                    sequence_ids.append(int(pred_np[batch_index, time_index]))
            all_pred_indices.append(sequence_ids)

    tag_sequences: list[list[str]] = []
    for sentence_index, tag_ids_row in enumerate(all_pred_indices):
        tokens_row = token_sents[sentence_index]
        predictions = [id2tag[idx] for idx in tag_ids_row]
        if len(predictions) < len(tokens_row):
            predictions.extend(["O"] * (len(tokens_row) - len(predictions)))
        elif len(predictions) > len(tokens_row):
            predictions = predictions[: len(tokens_row)]
        tag_sequences.append(predictions)

    write_predictions(output_path, token_sents, tag_sequences)
    print(f"已写入预测: {Path(output_path).resolve()} 句子数={len(token_sents)}")
