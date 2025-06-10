import os
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from torch import IntTensor, Tensor, cuda
import re

def getSequence(file_dir):
    f = open(file_dir, 'r')
    sequences = f.read().split('\n')
    f.close()
    return sequences

def getDataPair(posSeq, negSeq):
    seqs, labels = [], []
    for seq in posSeq:
        if seq.strip() != '':
            seqs.append(seq)
            labels.append(1)
    for seq in negSeq:
        if seq.strip() != '':
            seqs.append(seq)
            labels.append(0)
    return seqs, labels

def getAFP_Main_train(file_dir):
    df = pd.read_csv(f'{file_dir}/train.csv')
    seqs = df['seqs'].to_list()
    labels = df['labels'].to_list()
    features = df.iloc[:, 1:-1]
    return seqs, labels, features

def getAFP_Main_test(file_dir):
    df = pd.read_csv(f'{file_dir}/test.csv')
    seqs = df['seqs'].to_list()
    labels = df['labels'].to_list()
    features = df.iloc[:, 1:-1]
    return seqs, labels, features

def getAFP_Main(file_dir):
    train_seqs, train_labels, train_features = getAFP_Main_train(file_dir)
    test_seqs, test_labels, test_features = getAFP_Main_test(file_dir)
    return train_seqs, train_labels, train_features, test_seqs, test_labels, test_features
    
def encode(seqs, tokenizer, max_length):
    lengths = [len(seq) for seq in seqs]
    seqs = list(map(lambda x:" ".join(list(re.sub(r"[UZOB]", "X", x))), seqs))
    inputs = tokenizer(seqs, max_length=max_length, padding="max_length", return_tensors="pt")
    return inputs, lengths

def getDataloader(inputs, lengths, labels, train_idx=None, 
                  test_idx=None, device='cpu', batch_size=32, shuffle=False, sampler=None):
    if train_idx is None:
        dataset = TensorDataset(
            Tensor(inputs['input_ids']).to(device), 
            Tensor(inputs['attention_mask']).to(device),
            Tensor(lengths).to(device),
            Tensor(labels).to(device)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    else:
        train_dataset = TensorDataset(
            Tensor(inputs['input_ids'])[train_idx].to(device), 
            Tensor(inputs['attention_mask'])[train_idx].to(device),
            Tensor(lengths)[train_idx].to(device),
            Tensor(labels)[train_idx].to(device)
        )
        test_dataset = TensorDataset(
            Tensor(inputs['input_ids'])[test_idx].to(device), 
            Tensor(inputs['attention_mask'])[test_idx].to(device),
            Tensor(lengths)[test_idx].to(device),
            Tensor(labels)[test_idx].to(device)
        )
        return (DataLoader(train_dataset, batch_size=batch_size, 
                           shuffle=shuffle, sampler=sampler),
                DataLoader(test_dataset, batch_size=batch_size))

def getDataloaderwithFeature(inputs, lengths, features, labels, train_idx=None, 
                  test_idx=None, device='cpu', batch_size=32, shuffle=False, sampler=None):
    if train_idx is None:
        dataset = TensorDataset(
            Tensor(inputs['input_ids']).to(device), 
            Tensor(inputs['attention_mask']).to(device),
            Tensor(lengths).to(device),
            Tensor(features).to(device),
            Tensor(labels).to(device)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    else:
        train_dataset = TensorDataset(
            Tensor(inputs['input_ids'])[train_idx].to(device), 
            Tensor(inputs['attention_mask'])[train_idx].to(device),
            Tensor(lengths)[train_idx].to(device),
            Tensor(features)[train_idx].to(device),
            Tensor(labels)[train_idx].to(device)
        )
        test_dataset = TensorDataset(
            Tensor(inputs['input_ids'])[test_idx].to(device), 
            Tensor(inputs['attention_mask'])[test_idx].to(device),
            Tensor(lengths)[test_idx].to(device),
            Tensor(features)[test_idx].to(device),
            Tensor(labels)[test_idx].to(device)
        )
        return (DataLoader(train_dataset, batch_size=batch_size, 
                           shuffle=shuffle, sampler=sampler),
                DataLoader(test_dataset, batch_size=batch_size))

def transpose_data(data):
    length1 = len(data[0])
    length2 = len(data)
    data_t = []
    for i in range(length1):
        data_t.append([])
        for j in range(length2):
            data_t[-1].append(data[j][i])
    return data_t

def get_weights(labels):
    negative_weights = sum(labels) / len(labels)
    positive_weights = 1 - negative_weights
    weights = []
    for label in labels:
        if label == 0:
            weights.append(negative_weights)
        else:
            weights.append(positive_weights)
    return weights
