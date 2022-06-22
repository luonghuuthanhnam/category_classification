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


class Prettify_Text():
    def __init__(self, special_vocabs_path = "./special_vocabs.txt"):
        with open(special_vocabs_path, "r") as f:
            self.special_vocabs = f.readlines()
            self.special_vocabs = [each.lower().strip() for each in self.special_vocabs]
    def __call__(self, text: str) -> str:
        special_vocabs = self.special_vocabs
        text = text.lower().strip()
        # Remove punctuation
        punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        # Remove words including both letter(s) and digit(s). Note: DON'T remove words like PO1234
        pattern_punctuation = f"([{punctuation}])"
        pattern_word = r'''(\b(?!PO)[A-Z\p{L}]+[*\d@.,/-]+[\w@]*|[*\d@.,/-]+[A-Z\p{L}]+[\w@]*)'''
        prettified_list = []
        words = text.split()
        for word in words:
            if word not in special_vocabs:
                _sub_words = []
                for each in word.split():
                    _sub_word = re.sub(pattern_punctuation, " ", each)
                    _sub_word = re.sub(pattern_word, ' ', _sub_word, flags=re.IGNORECASE)
                    _sub_word = re.sub(r"\b\d{1,4}\b", " ", _sub_word)
                    _sub_words.append(_sub_word)
                _word = " ".join(_sub_words)
                prettified_list.append(_word)
            else:
                prettified_list.append(word)
        prettified_text = " ".join(prettified_list)

        # Remove words with less than 4 digits and redundant spaces
        prettified_text = re.sub("\s\s+" , " ", prettified_text)

        return prettified_text.lower().strip()
prettify_text = Prettify_Text(special_vocabs_path = "./special_vocabs.txt")