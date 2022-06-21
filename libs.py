import json
import random
import regex as re
import numpy as np
import torch
import pandas as pd
import transformers
from typing import List, Tuple
from collections import defaultdict, deque
from vncorenlp import VnCoreNLP
from tqdm import tqdm_notebook
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Dataset,
)
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
)
from fastprogress import master_bar, progress_bar
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import sys