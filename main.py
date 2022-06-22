from libs import *
from config import *
from utils import get_data_from_json, train_one_epoch, evaluate
from dataset_builder import CustomDataset

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-epochs",
        "--epochs",
        type=int,
        default=25,
        help="number of epochs",
    )
    ap.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=2e-5,
        help="learning rate",
    )
    ap.add_argument(
        "-begin",
        "--beginning_epoch",
        type=int,
        default=0,
        help="which epoch the training progress begins from",
    )
    ap.add_argument(
        "-use_pretrained",
        "--use_pretrained",
        choices=("0","1"),
        default="0",
        help="Use checkpoint of masked language model",
    )
    ap.add_argument(
        "-d",
        "--destination_path",
        type=str,
        help="model destination path",
    )
    ap.add_argument(
        "-use_new_tokens",
        "--use_new_tokens",
        choices=("0","1"),
        default="0",
        help="New tokens have been added to tokenizer",
    )
    args = vars(ap.parse_args())
    (
        epochs,
        lr,
        beginning_epoch,
        use_pretrained,
        model_destination_path,
        use_new_tokens,
    ) = (
        args["epochs"],
        args["learning_rate"],
        args["beginning_epoch"],
        bool(int(args["use_pretrained"])),
        args["destination_path"],
        bool(int(args["use_new_tokens"])),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # DECLARE TOKENIZER
    tokenizer_checkpoint = f"./tokenizers/{'new-tokens-added' if  use_new_tokens else 'original'}/tokenizer"
    print(tokenizer_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

    # DECLARE SEQUENCE CLASSIFICATION MODEL
    # model_checkpoint = "./checkpoints/original/phoBERT_base" 
    model = AutoModelForSequenceClassification.from_pretrained(SEQ_CLS_MODEL_CHECKPOINT)

    print(f" \
    Tokenizer checkpoint: {tokenizer_checkpoint} \
    Model checkpoint: {SEQ_CLS_MODEL_CHECKPOINT} \
    Use pretrained: {use_pretrained} \
    Use new tokens: {use_new_tokens} \
    ")

    if use_pretrained:
        mlm_checkpoint = f"./mlm-checkpoint-{'new-tokens' if  use_new_tokens else 'no-new-tokens'}"
        print(f"MaskedLM checkpoint: {mlm_checkpoint}")
        mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_checkpoint)
        model.roberta = mlm_model.roberta

    model.to(device)

    # LOAD DATA FROM DISK
    train_sentences, train_labels = get_data_from_json(TRAIN_SET_FOR_CATEGORY_CLS)
    train_labels = list(map(lambda label: model.config.label2id[label], train_labels))
    val_sentences, val_labels = get_data_from_json(VAL_SET_FOR_CATEGORY_CLS)
    val_labels = list(map(lambda label: model.config.label2id[label], val_labels))

    # BUILD DATASET
    train_dataset = CustomDataset(
        train_sentences, train_labels, MAX_LEN, tokenizer, VNCORENLP_JAR_PATH, model.config.label2id["Other"]
    )
    val_dataset = CustomDataset(
        val_sentences, val_labels, MAX_LEN, tokenizer, VNCORENLP_JAR_PATH, model.config.label2id["Other"]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.05,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    class_weights = None
    if USE_CLASS_WEIGHT:
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels
        )
        class_weights = torch.from_numpy(class_weights).float().to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    learning_log = []
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    for epoch in range(beginning_epoch, beginning_epoch + epochs):
        epoch_train_loss, epoch_train_acc, f1_train_score = train_one_epoch(
            model, train_dataloader, loss_function, optimizer, None, device
        )
        epoch_val_loss, epoch_val_acc, f1_val_score = evaluate(
            model, val_dataloader, loss_function, device
        )
        learning_log.append(
            {
                "epoch": epoch,
                "train loss": np.round(epoch_train_loss, 4),
                "val loss": np.round(epoch_val_loss, 4),
                "train accuracy": np.round(epoch_train_acc, 4),
                "val accuracy": np.round(epoch_val_acc, 4),
                "train f1_score": np.round(f1_train_score, 4),
                "val f1_score": np.round(f1_val_score, 4),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if epoch % 2 == 0 and epoch >= 14:
            scheduler.step()
        print(learning_log[-1])

    print(learning_log)
    # if model_destination_path:

    model_destination_path = "./checkpoints/2022-06-22"
    model.save_pretrained(model_destination_path)
    torch.save(optimizer.state_dict(), f"./checkpoints/optimizer.pth")

    sys.exit()
