import pandas as pd 
import torch
from torch.optim import Adam

from sklearn.model_selection import train_test_split

from config import config
from src.dataloader import bertDataloader
from src.train import train
from src.model import BertBaseUncased


def trainer():

    df = pd.read_csv(config.TRAINING_FILE).fillna(0)
    df = df.iloc[:100]
    print("shape of training data: ", df.shape)
    df["sentiment"] = df["sentiment"].apply(
        lambda x: 1 if x=="positive" else 0
    )

    train_data, val_data = train_test_split(df, test_size= 0.2, stratify= df["sentiment"])
    
    train_dataloader = bertDataloader(
        train_data["review"].values,
        train_data["sentiment"].values
    )
    val_dataloader = bertDataloader(
        val_data["review"].values,
        val_data["sentiment"].values
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataloader,
        config.TRAIN_BATCH_SIZE
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataloader,
        config.VAL_BATCH_SIZE
    )

    model = BertBaseUncased()

    optimizer = Adam(model.parameters(), lr= 3e-5)

    for epoch in range(config.NEPOCHS):
        train(
            model= model,
            optimizer= optimizer,
            train_dataloader = train_dataloader

        )
    


if __name__ == "__main__":
    trainer()