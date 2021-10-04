from tqdm import tqdm
import torch.nn as nn
import torch
from datetime import datetime


def train_model(model, epochs, train_dataloader, l_rate=0.001, 
                loss_fn=nn.BCEWithLogitsLoss(), clip_val=2):

    criterion = loss_fn
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
    clip_value = clip_val

    # set train mode
    model.train()
    # perform training for a set number of epochs
    for epoch in range(epochs):
        # status bar
        loop = tqdm(train_dataloader, total=len(train_dataloader), leave=True)
        # iterate through batches in training dataloader
        for batch in loop:
            # predict outputs for a batch
            output = model(batch)
            optimizer.zero_grad()
            # get gold labels for comparison
            labels = [sample['binary_label'] for sample in batch]
            # define loss
            # target as tensor for loss function
            loss_target = torch.tensor(labels)
            loss = criterion(output.float(), loss_target.float())
            loss.requires_grad = True
            # set backward update
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value)
            optimizer.step()
            # progress description
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

    # save model using current timestamp
    timestamp = str(datetime.now())[:10] + '_' + str(datetime.now())[11:16]
    timestamp = timestamp.replace(':', '-')
    model_path = f'saved_models/saved_model_{timestamp}'
    print(f'Model saved.\nLocation:{model_path}\nTimestamp: {timestamp}')
    torch.save(model.state_dict(), model_path)

    return model.state_dict() 
