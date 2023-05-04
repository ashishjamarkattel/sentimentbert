import transformers

NEPOCHS = 1
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
TRAINING_FILE = "data\IMDB Dataset.csv"
BERT_PATH = "bert"
MODEL = "bert\pytorch_model.bin"
MAX_LENGTH = 128
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH
)