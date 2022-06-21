from libs import *
from config import *


def get_data_from_json(json_path: str) -> Tuple[List[str], List[str]]:
    with open(json_path) as json_file:
        dataset = json.load(json_file)
    sentences = [each["itemName"] for each in dataset]
    labels = [each["label"] for each in dataset]
    return sentences, labels


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)
    return accuracy, f1_score(labels_flat, pred_flat, average="weighted")


def train_one_epoch(
    model, dataloader, loss_function, optimizer, master_bar=None, device="cpu"
) -> Tuple[float, float, float]:
    model.train()
    epoch_train_loss, epoch_train_acc, f1_train_score = list(), list(), list()
    for _, batch in enumerate(tqdm(dataloader)):
    # for batch in progress_bar(dataloader, parent=master_bar):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        model.zero_grad()
        outputs = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )
        logits = outputs[1]
        loss = loss_function(logits, b_labels) if USE_CLASS_WEIGHT else outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()
        tmp_train_accuracy, tmp_train_f1 = flat_accuracy(logits, label_ids)

        epoch_train_loss.append(loss.item())
        epoch_train_acc.append(tmp_train_accuracy)
        f1_train_score.append(tmp_train_f1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # master_bar.child.comment = (
        #     "train loss: %f, train accuracy: %f, f1 score: %f"
        #     % (
        #         np.mean(epoch_train_loss),
        #         np.mean(epoch_train_acc),
        #         np.mean(f1_train_score),
        #     )
        # )

    return np.mean(epoch_train_loss), np.mean(epoch_train_acc), np.mean(f1_train_score)


def evaluate(model, dataloader, loss_function=None, device="cpu"):
    model.eval()
    epoch_val_loss, epoch_val_acc, f1_val_score = list(), list(), list()
    for _, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )
            logits = outputs[0]
            if loss_function:
                loss = loss_function(logits, b_labels)
                epoch_val_loss.append(loss.item())

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy()

            tmp_eval_accuracy, tmp_eval_f1 = flat_accuracy(logits, label_ids)

            epoch_val_acc.append(tmp_eval_accuracy)
            f1_val_score.append(tmp_eval_f1)

    mean_val_loss = np.mean(epoch_val_loss) if len(epoch_val_loss) else None
    return mean_val_loss, np.mean(epoch_val_acc), np.mean(f1_val_score)


def prettify_text(text: str) -> str:
    text = text.lower().strip()
    # Remove words including both letter(s) and digit(s). Note: DON'T remove words like PO1234
    pattern = r'''(\b(?!PO)[A-Z\p{L}]+[*\d@.,/-]+[\w@]*|[*\d@.,/-]+[A-Z\p{L}]+[\w@]*)'''
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Remove punctuation
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    pattern = f"([{punctuation}])"
    text = re.sub(pattern, " ", text)

    # Remove words with less than 4 digits and redundant spaces
    text = re.sub(r"\b\d{1,4}\b|\s{2,}", " ", text)
    return text.lower().strip()