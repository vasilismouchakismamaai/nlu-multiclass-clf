import os
import torch

# CONSTANTS
DATA_PATH = "data/army-intents.csv"
# DATA_PATH = "data/train.csv"
# DATA_PATH = "/content/drive/MyDrive/nlp/datasets/army-intents.csv"

MODELS_PATH = os.path.join(os.getcwd(), "models")

BERT_MODEL_NAME = "bert-base-multilingual-cased"
# BERT_MODEL_NAME = "sentence-transformers/paraphrase-mpnet-base-v2"
MAX_TOKEN_COUNT = 128
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 5e-04
THRESHOLD = 0.5

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"On `{DEVICE}` device.")