import torch

def train(
    model,
    optimizer,
    train_dataloader
):
    model.train()

    for i, data in enumerate(train_dataloader):

        ids = data["ids"]
        attention_mask = data["mask"]
        token_type_ids = data["token_type_ids"]
        
        target = data["target"]

        output = model(
            ids = ids,
            mask = attention_mask,
            token_type_ids = token_type_ids
        )
        
        loss = torch.nn.BCEWithLogitsLoss()(output.squeeze(0), target)
        loss.backward()
        optimizer.step()

