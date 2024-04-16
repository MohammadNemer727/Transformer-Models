import numpy as np
import pandas as pd
import torch
import random
import os
from numpy import genfromtxt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import math
from torch import nn
from sklearn.model_selection import train_test_split
import copy
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torchinfo import summary
import torch.nn.functional as F 
from einops import rearrange

class EcgDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.X = torch.tensor(x)
        self.Y = torch.tensor(y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index].unsqueeze(1), self.Y[index]

class MyTestDataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        file_out_test = pd.read_csv('/content/drive/MyDrive/dataset/mitbih_train.csv')
        x_test = file_out_test.iloc[:, :-1].values
        y_test = file_out_test.iloc[:, -1:].astype(dtype=int).values
        test_set = EcgDataset(x=x_test, y=y_test)
        self.dataLoader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True)

    def getDataLoader(self):
        return self.dataLoader

class myDataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        no_points = 1000
        train_data = np.empty((0, no_points * 8), float)
        test_data = np.empty((0, no_points * 8), float)
        train_label = []
        test_label = []

        folder_dir = './data/full'
        classes = os.listdir(folder_dir)
        train_counter = 0

        for class_name in classes:
            class_path = os.path.join(folder_dir, class_name)
            chosen_files = []
            k = 2  # Number of files to choose randomly from each class
            for _ in range(k):
                csv_file = random.choice(os.listdir(class_path))
                while csv_file in chosen_files:
                    csv_file = random.choice(os.listdir(class_path))
                chosen_files.append(csv_file)
                print("loading ", class_path, " ", csv_file)
                try:
                    file_data = genfromtxt(os.path.join(class_path, csv_file), delimiter=',')
                    for i in range(0, len(file_data), no_points):
                        segment = file_data[i:i + no_points, :].reshape(-1)
                        if train_counter < 4:
                            train_data = np.vstack([train_data, segment])
                            train_label.append(class_name)
                            train_counter += 1
                        else:
                            test_data = np.vstack([test_data, segment])
                            test_label.append(class_name)
                            train_counter = 0
                except Exception as e:
                    print(f"Error processing file {csv_file}: {e}")

        # Encoding labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(train_label)

        label_encoder = LabelEncoder()
        y_test_encoded = label_encoder.fit_transform(test_label)

        train_set = EcgDataset(x=train_data, y=y_train_encoded)
        val_set = EcgDataset(x=test_data, y=y_test_encoded)

        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_set, batch_size=batch_size, shuffle=True)
        }
        self.dataloaders = dataloaders

    def getDataLoader(self):
        return self.dataloaders

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class PostionalEncoding(nn.Module):
    def __init__(self, dropout=0.1, max_seq_len=5000, d_model=512, batch_first=False):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(self.x_dim)]
        x = self.dropout(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, d_model, seq_len, details, n_classes=6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.details = details
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * seq_len, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.seq(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class ScaleDotProductAttention(nn.Module):
    def __init__(self, details):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.details = details

    def forward(self, q, k, v, e=1e-12):
        batch_size, head, length, d_tensor = k.size()
        score = (q @ k.transpose(2, 3)) / math.sqrt(d_tensor)
        score = self.softmax(score)
        v = score @ v
        return v, score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, details):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention(details=details)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.details = details

    def forward(self, q, k, v):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v)
        
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, details, device):
        super(EncoderLayer, self).__init__()
        self.prob_attn = ProbSparseSelfAttention(d_model, n_head, drop_prob, device)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.details = details

    def forward(self, x):
        _x = x
        x = self.prob_attn(x)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob, details, device):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden, n_head, drop_prob, details, device) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, drop_prob, device):
        super(ProbSparseSelfAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.device = device
        self.scale = (d_model // n_head) ** -0.5
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b s (h d) -> b h s d', h=self.n_head), (q, k, v))
        print("Shapes: q={}, k={}, v={}".format(q.shape, k.shape, v.shape))  # Debugging statement
        scores = torch.einsum('bhid,bhjd->bhij', q * self.scale, k)
        mask = torch.bernoulli(torch.ones((batch_size, self.n_head, seq_len), device=self.device) * 0.5).bool()
        scores.masked_fill_(~mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.einsum('bhij,bhjd->bhid', attn, v)
        context = rearrange(context, 'b h s d -> b s (h d)')
        output = self.proj(context)
        return output


class Informer(nn.Module):
    def __init__(self, device, d_model=100, n_head=4, max_len=5000, seq_len=200, ffn_hidden=128, n_layer=2, drop_prob=0.1, details=False):
        super(Informer, self).__init__()
        self.device = device
        self.encoder_input_layer = nn.Linear(sequence_len, d_model)
        self.encoder = Encoder(d_model, ffn_hidden, n_head, n_layer, drop_prob, details, device)
        self.classHead = ClassificationHead(seq_len, d_model, details)
        
    def forward(self, src):
        if src.dim() == 2:
            src = src.unsqueeze(-1) 
       
        src = self.encoder_input_layer(src)
        enc_src = self.encoder(src)
        cls_res = self.classHead(enc_src)
        return cls_res


def cross_entropy_loss(pred, target):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred, target.squeeze(-1))
    return loss

def calc_loss_and_score(pred, target, metrics):
    softmax = nn.Softmax(dim=1)
    pred = pred.squeeze(-1)
    target = target.squeeze(-1).long()
    ce_loss = cross_entropy_loss(pred, target)
    metrics['loss'].append(ce_loss.item())
    pred = softmax(pred)
    _, pred = torch.max(pred, dim=1)
    metrics['correct'] += torch.sum(pred == target).item()
    metrics['total'] += target.size(0)
    return ce_loss

def print_metrics(main_metrics_train, main_metrics_val, metrics, phase):
    correct = metrics['correct']
    total = metrics['total']
    accuracy = 100 * correct / total
    loss = metrics['loss']
    if phase == 'train':
        main_metrics_train['loss'].append(np.mean(loss))
        main_metrics_train['accuracy'].append(accuracy)
    else:
        main_metrics_val['loss'].append(np.mean(loss))
        main_metrics_val['accuracy'].append(accuracy)
    result = "phase: " + str(phase) + \
               ' \nloss : {:4f}'.format(np.mean(loss)) + ' accuracy : {:4f}'.format(accuracy) + "\n"
    return result
def train_model(dataloaders, model, optimizer, num_epochs=100):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    main_metrics_train = {'loss': [], 'accuracy': []}
    main_metrics_val = {'loss': [], 'accuracy': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            metrics = {'loss': [], 'correct': 0, 'total': 0}

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.int)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss_and_score(outputs, labels, metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            print(print_metrics(main_metrics_train=main_metrics_train, main_metrics_val=main_metrics_val, metrics=metrics, phase=phase))
            epoch_loss = np.mean(metrics['loss'])

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    return model

device = torch.device("cpu")
sequence_len = 2000
max_len = 5000
n_head = 2
n_layer = 1
drop_prob = 0.1
d_model = 128
ffn_hidden = 128
batch_size = 4

model = Informer(device, d_model, n_head, max_len, sequence_len, ffn_hidden, n_layer, drop_prob, False).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataloaders = myDataLoader(batch_size).getDataLoader()
model_normal_ce = train_model(dataloaders, model, optimizer, num_epochs=20)
torch.save(model.state_dict(), 'informer')
