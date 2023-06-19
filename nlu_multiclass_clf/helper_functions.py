import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# helper functions
def create_one_hot_dataset(df, labels_to_drop):
    tmp = pd.get_dummies(df.label, dtype=int)
    data = pd.concat([df, tmp], axis=1)
    # drop `_garbage` label, off-topic text should have all labels equal to 0
    data = data.drop(labels_to_drop, axis=1)
    return data

def split_train_val_test(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)#, stratify=df["label"])
    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)#, stratify=train_df["label"])
    return train_df, valid_df, test_df

def calculate_accuracy(predictions, labels):
    threshold = 0.7
    predicted_labels = (predictions > threshold).int()
    correct_predictions = (predicted_labels == labels).all(dim=1).sum().item()
    accuracy = correct_predictions / labels.size(0)
    return accuracy

def compute_metrics(predicted_labels_list, true_labels_list):
    predicted_labels = torch.cat(predicted_labels_list, dim=0)
    true_labels = torch.cat(true_labels_list, dim=0)

    # Convert tensors to numpy arrays
    predicted_labels = predicted_labels.cpu().numpy()
    true_labels = true_labels.cpu().numpy()
    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='micro')
    recall = recall_score(true_labels, predicted_labels, average='micro')
    f1 = f1_score(true_labels, predicted_labels, average='micro')
    # print(accuracy, precision, recall, f1)
    return accuracy, precision, recall, f1