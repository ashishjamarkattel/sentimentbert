import transformers
import torch.nn as nn
import torch

import os

from config import config

class BertBaseUncased(nn.Module):

    def __init__(self):
        super(BertBaseUncased, self).__init__()
        self.model = transformers.BertModel.from_pretrained(
            config.BERT_PATH,
            return_dict= False
        )  
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):

        _, cls_ = self.model(
            input_ids = ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        do = self.dropout(cls_)
        output = self.fc1(do)

        return output
        