import transformers
import torch

from config import config

class bertDataloader:
    def __init__(self, review, sentiment):
        self.review = review
        self.target = sentiment
        self.max_length = config.MAX_LENGTH

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        inputs = config.TOKENIZER.encode_plus(
            review,
            None,
            max_length = self.max_length,
            padding= "max_length",
            truncation= True
        )
        input_ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        
        return {
            "ids" : torch.tensor(input_ids),
            "mask": torch.tensor(mask),
            "token_type_ids": torch.tensor(token_type_ids),
            "target": torch.tensor(self.target[item], dtype=torch.float)
        }