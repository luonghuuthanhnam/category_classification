from libs import *

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

MODEL_CHECKPOINT = "vinai/phobert-base"
# SEQ_CLS_MODEL_CHECKPOINT = "./checkpoints/original/phoBERT_base"
SEQ_CLS_MODEL_CHECKPOINT = "./pretrain_added_new_token"
TOKENIZER_CHECKPOINT = "./tokenizer_added_new_token"

DATASET_FOR_CATEGORY_CLS = (
    "./dataset/2022-06-27/dataset_for_category_classification.json"
)
TRAIN_SET_FOR_CATEGORY_CLS = "./dataset/2022-06-27/seq_cls/train.json"
VAL_SET_FOR_CATEGORY_CLS = "./dataset/2022-06-27/seq_cls/val.json"


VNCORENLP_JAR_PATH = "./vncorenlp/VnCoreNLP-1.1.1.jar"
MAX_LEN = 125
BATCH_SIZE = 32
USE_CLASS_WEIGHT = True
