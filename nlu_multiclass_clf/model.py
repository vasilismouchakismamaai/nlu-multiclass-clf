from torch import nn
from torch import Tensor
from transformers import BertModel, BertTokenizerFast
from config import BERT_MODEL_NAME


class MultiClassModel(nn.Module):
    def __init__(self, n_labels: int, dropout: float = 0.1) -> None:
        super(MultiClassModel, self).__init__()
        self.model = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, n_labels)
        # self.bn = nn.BatchNorm1d(n_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, mask) -> Tensor:
        out = self.model(input_ids, attention_mask=mask)
        out = out.pooler_output
        out = self.dropout(out)
        out = self.linear(out)
        # out = self.bn(out)
        out = self.sigmoid(out)

        return out
