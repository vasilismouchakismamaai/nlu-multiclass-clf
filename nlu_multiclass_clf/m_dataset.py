import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from config import BERT_MODEL_NAME


class MultiClassDataset(Dataset):
    def __init__(self, data: pd.DataFrame, max_token_len: int = 128):
        self.tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
        self.data = data
        self.max_token_len = max_token_len
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text = data_row.text
        labels = data_row[self.data.columns.tolist()[2:]]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            # token_type_ids=encoding["token_type_ids"].flatten(),
            labels=torch.FloatTensor(labels)
        )