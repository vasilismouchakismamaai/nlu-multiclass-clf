import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import *
from helper_functions import *
from model import MultiClassModel
from m_dataset import MultiClassDataset

import warnings
warnings.filterwarnings("ignore")


def main():
    data = pd.read_csv(DATA_PATH)
    data = create_one_hot_dataset(data, ["_garbage"])
    train_data, valid_data, test_data = split_train_val_test(data)
    
    # create pytorch datasets
    train_dataset = MultiClassDataset(train_data, max_token_len=MAX_TOKEN_COUNT)
    valid_dataset = MultiClassDataset(valid_data, max_token_len=MAX_TOKEN_COUNT)
    test_dataset = MultiClassDataset(test_data, max_token_len=MAX_TOKEN_COUNT)

    ###########
    # check if labels are correct - they are !
    # sample = train_dataset[56]
    # lbl = sample["label"]
    # print(lbl)
    # print(data.columns.tolist()[2:][sample["labels"].argmax()])
    # exit()
    ###########

    # create pytorch dataloaders
    train_loader = DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, VALID_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, VALID_BATCH_SIZE, shuffle=False)

    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_data.shape))
    print("VALID Dataset: {}".format(valid_data.shape))
    print("TEST Dataset: {}".format(test_data.shape))
    print(len(data.columns.tolist()[2:]))

    n_labels = len(data.columns.tolist()[2:])
    model = MultiClassModel(n_labels).to(DEVICE)

    criterion = nn.BCELoss() # because we apply sigmoid inside the model
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE),
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE),
    # optimizer = torch.optim.Adadelta(model.parameters()),
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_loader) * EPOCHS
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # Warm-up for 10% of total steps
        num_training_steps=total_steps
    )

    # ---------- TRAIN LOOP ------------
    train_loss = 0
    valid_loss = 0
    train_acc = 0
    valid_acc = 0
    val_targets = []
    val_outputs = []
    
    train_predicted_labels = []
    train_true_labels = []
    valid_predicted_labels = []
    valid_true_labels = []

    for epoch in tqdm(range(EPOCHS), desc="Epochs "):
        total_acc_train, total_loss_train = 0, 0
        model.train()
        train_correct = 0
        train_total = 0
        # for batch_idx, batch in tqdm(train_loader, total=len(train_loader), desc="Batches ", leave=False):
        # for batch_idx, batch in enumerate(train_loader, 0):
        train_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Train batches ", leave=False)):
            ids = batch["input_ids"].to(DEVICE, dtype=torch.long)
            mask = batch["attention_mask"].to(DEVICE, dtype=torch.long)
            # token_type_ids = batch["token_type_ids"].to(DEVICE, dtype=torch.long)
            targets = batch["labels"].to(DEVICE, dtype=torch.float)
            outputs = model(ids, mask)

            max_indices = torch.argmax(outputs, dim=1)
            predicted_labels = torch.zeros_like(outputs)
            predicted_labels[torch.arange(outputs.size(0)), max_indices] = 1
            
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            # loss = criterion(predicted_labels.requires_grad_(), targets)
            train_loss += loss.item()

            # max_indices = torch.argmax(outputs, dim=1)
            # predicted_labels = torch.zeros_like(outputs)
            # predicted_labels[torch.arange(outputs.size(0)), max_indices] = 1
            # predicted_labels = (outputs > THRESHOLD).float()
            train_predicted_labels.append(predicted_labels)
            train_true_labels.append(targets)

            loss.backward()
            optimizer.step()
        scheduler.step()
        
        t_acc, t_precision, t_recall, t_f1 = compute_metrics(train_predicted_labels, train_true_labels)

        model.eval()
        total_acc_val, total_loss_val = 0, 0
        valid_correct = 0
        valid_total = 0
        valid_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(valid_loader, desc="Validation batches ", leave=False)):
                ids = batch["input_ids"].to(DEVICE, dtype=torch.long)
                mask = batch["attention_mask"].to(DEVICE, dtype=torch.long)
                # token_type_ids = batch["token_type_ids"].to(DEVICE, dtype=torch.long)
                targets = batch["labels"].to(DEVICE, dtype=torch.float)
                outputs = model(ids, mask)

                max_indices = torch.argmax(outputs, dim=1)
                predicted_labels = torch.zeros_like(outputs)
                predicted_labels[torch.arange(outputs.size(0)), max_indices] = 1

                loss = criterion(outputs, targets)
                # loss = criterion(predicted_labels.requires_grad_(), targets)
                valid_loss += loss.item()

                # predicted_labels = (outputs > THRESHOLD).float()
                valid_predicted_labels.append(predicted_labels)
                valid_true_labels.append(targets)

                # valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                # val_targets.extend(targets.cpu().detach().numpy().tolist())
                # val_outputs.extend(outputs.cpu().detach().numpy().tolist())

        v_acc, v_precision, v_recall, v_f1 = compute_metrics(valid_predicted_labels, valid_true_labels)

        train_loss = train_loss/len(train_loader)
        valid_loss = valid_loss/len(valid_loader)
        # print training/validation statistics 
        print('\nEpoch: {} \nAvgerage Training Loss: {:.6f} \
        \nAverage Validation Loss: {:.6f} \
        \nTrain Acc: {:.6f} \
        \nTrain Precision: {:.6f} \
        \nTrain Recall: {:.6f} \
        \nTrain F1: {:.6f} \
        \nValid Acc: {:.6f} \
        \nValid Precision: {:.6f} \
        \nValid Recall: {:.6f} \
        \nValid F1: {:.6f}'.format(
            epoch+1, 
            train_loss / len(train_loader),
            valid_loss / len(train_loader),
            t_acc,
            t_precision,
            t_recall,
            t_f1,
            v_acc,
            v_precision,
            v_recall,
            v_f1
            ))

    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    torch.save(model, os.path.join(MODELS_PATH, "model.pt"))
    torch.save(model.state_dict(), os.path.join(MODELS_PATH, "model_dict.pt"))

if __name__ == "__main__":
    main()
