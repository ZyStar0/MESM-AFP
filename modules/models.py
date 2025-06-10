import re
import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel
from transformers import EsmModel, EsmTokenizer
from transformers import AutoModel
from transformers import BertForMaskedLM

class sequenceClassifier(nn.Module):
    def __init__(self, model_name, classifier_name, classifier_args=tuple()):
        super(sequenceClassifier, self).__init__()
        self.embeds = Bert(model_name)
        self.classifier = globals()[classifier_name.upper()](*classifier_args)
    
    def forward(self, batch):
        embedding_repr = self.embeds(*batch)
        return self.classifier(embedding_repr)

    def getEmbeds(self, batch):
        return self.embeds(*batch)

class sequenceClassifierwithFeature(nn.Module):
    def __init__(self, model_name, classifier_name, classifier_args, feature_nums):
        super(sequenceClassifierwithFeature, self).__init__()
        self.embeds = Bert(model_name)
        self.classifier = globals()[classifier_name.upper()](classifier_args[0] + feature_nums, *classifier_args[1:])
    def forward(self, batch, features):
        embedding_repr = self.embeds(*batch)
        return self.classifier(torch.cat((embedding_repr, features), dim=1))
    
    def getEmbeds(self, batch):
        return self.embeds(*batch)
    
class Bert(nn.Module):
    def __init__(self, model_name):
        super(Bert, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask, lengths):
        embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = embedding_repr.last_hidden_state * attention_mask.unsqueeze(-1)
        return torch.sum(last_hidden_state, dim=1) / lengths.unsqueeze(1).to(torch.float32)
    

class Esm(nn.Module):
    def __init__(self, model_name='facebook/esm2_t30_150M_UR50D'):
        super(Esm, self).__init__()
        self.model = EsmModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, lengths):
        embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = embedding_repr.last_hidden_state * attention_mask.unsqueeze(-1)
        return torch.sum(last_hidden_state, dim=1) / lengths.unsqueeze(1).to(torch.float32)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout=0.2):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, hidden_states):
        hidden_states = self.mlp(hidden_states)
        return torch.sigmoid(hidden_states)


class CNN(nn.Module):
    def __init__(self, hidden_size):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(hidden_size * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return F.sigmoid(self.fc3(x))

