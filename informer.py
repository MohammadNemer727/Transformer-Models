import numpy as np
import pandas as pd
import torch
import random
import os
from numpy import genfromtxt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import math
from torch import nn
import copy
from torch.optim import Adam
import time
from torch.nn import LayerNorm

class MyDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.X = torch.tensor(x)
        self.Y = torch.tensor(y)
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index): 
        return self.X[index].unsqueeze(1).float(), self.Y[index] 

class MyDataLoader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        no_points = 1000
        train_data = []
        test_data = []
        train_label = []
        test_label = []

        folder_dir = '/content/drive/MyDrive/full'
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
                            train_data.append(segment)
                            train_label.append(class_name)
                            train_counter += 1
                        else:
                            if len(segment) == len(train_data[-1]):
                                test_data.append(segment)
                                test_label.append(class_name)
                                train_counter = 0
                            else:
                                print("Skipping segment with inconsistent shape")
                except Exception as e:
                    print(f"Error processing file {csv_file}: {e}")

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(train_label)

        label_encoder = LabelEncoder()
        y_test_encoded = label_encoder.fit_transform(test_label)

        train_set = MyDataset(x=train_data, y=y_train_encoded) 
        val_set = MyDataset(x=test_data, y=y_test_encoded) 

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
    def __init__(
        self,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        d_model: int = 512,
        batch_first: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

    def forward(self, x):
        max_seq_len = x.size(self.x_dim)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(max_seq_len, 1, self.d_model, dtype=torch.float)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.to(x.device)
        x = x + pe
        x = self.dropout(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, seq_len, d_model, n_classes: int = 6):
        super().__init__()
        self.seq = nn.Sequential(nn.Flatten(), nn.Linear(d_model * seq_len , 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_classes))
 
    def forward(self, x):
        x = self.seq(x)

class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, sampling_factor=5):
        super(ProbSparseAttention, self).__init__()
        self.sampling_factor = sampling_factor

    def forward(self, q, K, V):
        K_sample = K[:, :8000]

        try:
            Q_K_sample = torch.bmm(q, K_sample.transpose(1, 2))
        except RuntimeError as e:
            print("Error during torch.bmm:", e)
            raise RuntimeError("Error during torch.bmm. Please check your input dimensions.")

        L_K = K_sample.size(1)
        L_Q = q.size(1)
        log_L_K = torch.ceil(torch.log1p(torch.tensor(L_K))).to(torch.int).item()
        log_L_Q = torch.ceil(torch.log1p(torch.tensor(L_Q))).to(torch.int).item()

        U_part = min(self.sampling_factor * L_Q * log_L_K, L_K)
        index_sample = torch.randint(0, L_K, (U_part,))
        K_sample = K[:, index_sample, :]

        if q.dim() == 2:
            q = q.unsqueeze(1)
            q = q.unsqueeze(0)
        if K_sample.dim() == 2:
            K_sample = K_sample.unsqueeze(0)

        try:
            Q_K_sample = torch.bmm(q, K_sample.transpose(1, 2))
        except RuntimeError as e:
            print("Error during torch.bmm:", e)
            raise RuntimeError("Error during torch.bmm. Please check your input dimensions.")

        M = Q_K_sample.max(dim=-1)[0] - torch.div(Q_K_sample.sum(dim=-1), L_K)
        u = min(self.sampling_factor * log_L_Q, L_Q)
        M_top = M.topk(u, largest=False)[1]
        dim_for_slice = torch.arange(q.size(0)).unsqueeze(-1)
        Q_reduce = q[dim_for_slice, M_top]
        d_k = q.size(-1)
        attn_scores = torch.bmm(Q_reduce, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.bmm(attn_probs, V)
        return attn_output, attn_scores

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, sampling_factor):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ProbSparseAttention(d_model=d_model, sampling_factor=sampling_factor)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)  

        batch_size, query_len, d_model = q.size()
        d_tensor = d_model // self.n_head
        q = q.view(batch_size, self.n_head, query_len, d_tensor)

        k, v = self.split(k), self.split(v)

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
    def __init__(self, d_model, ffn_hidden, n_head, sampling_factor, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head, sampling_factor=sampling_factor)
        self.norm1 = LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(normalized_shape=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        _x = x
        x = self.attention(q=x, k=x, v=x)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob, sampling_factor):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  sampling_factor=sampling_factor,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Informer(nn.Module):
    def __init__(self, device, d_model=100, n_head=4, max_len=5000, seq_len=200,
                 ffn_hidden=128, n_layers=2, drop_prob=0.1, sampling_factor=5):
        super().__init__()
        self.device = device
        self.encoder_input_layer = nn.Linear(
            in_features=1,
            out_features=d_model
        )
        self.pos_emb = PostionalEncoding(max_seq_len=max_len, batch_first=False, d_model=d_model, dropout=0.1)
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               sampling_factor=sampling_factor)
        self.classHead = ClassificationHead(seq_len=seq_len, d_model=d_model, n_classes=5)
        self.classHead2 = ClassificationHead(seq_len=seq_len, d_model=d_model, n_classes=2)

    def forward(self, x):
        x = self.encoder_input_layer(x)
        x = x.transpose(0, 1)
        x = self.pos_emb(x)
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.transpose(0, 1)
        y1 = self.classHead(x)
        y2 = self.classHead2(x)
        return y1, y2

def train_model(device, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs1, outputs2 = model(inputs)
                    loss1 = criterion(outputs1, labels)
                    loss2 = criterion(outputs2, labels)

                    loss = loss1 + loss2

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(torch.argmax(outputs1, 1) == labels.data) + torch.sum(torch.argmax(outputs2, 1) == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / (2 * dataset_sizes[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    dataloaders = MyDataLoader(batch_size=1).getDataLoader()
    dataset_sizes = {'train': 2400, 'val': 600} 

    model = Informer(device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    trained_model = train_model(device, model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=25)
