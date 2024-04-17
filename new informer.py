import numpy as np
import torch
import random
import os
from numpy import genfromtxt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import math
from torch import nn
import copy
from torch import optim

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.X = torch.tensor(x, dtype=torch.float32)
        self.Y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

class MyDataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        no_points = 5000
        train_data = np.empty((0, no_points * 8), float)
        test_data = np.empty((0, no_points * 8), float)
        train_label = []
        test_label = []

        folder_dir = 'C:/Users/manemer/Desktop/DL/full'
        classes = os.listdir(folder_dir)
        train_counter = 0

        for class_name in classes:
            class_path = os.path.join(folder_dir, class_name)
            chosen_files = []
            k = 15
            for _ in range(k):
                csv_file = random.choice(os.listdir(class_path))
                while csv_file in chosen_files:
                    csv_file = random.choice(os.listdir(class_path))
                chosen_files.append(csv_file)
                print("loading ", os.path.join(class_path, csv_file))
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
        y_test_encoded = label_encoder.transform(test_label)

        train_set = MyDataset(train_data, y_train_encoded)
        val_set = MyDataset(test_data, y_test_encoded)

        self.dataloaders = {
            'train': DataLoader(train_set, batch_size=self.batch_size, shuffle=True),
            'val': DataLoader(val_set, batch_size=self.batch_size, shuffle=True)
        }

    def get_data_loader(self):
        return self.dataloaders


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len=5000, d_model=512, drop_prob=0.1, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=drop_prob)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(self.x_dim)].to(x.device)
        x = self.dropout(x)
        return x

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        score = self.softmax(score)
        v = torch.matmul(score, v)
        return v, score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        out, attention = self.attention(q, k, v)

        out = self.concat_heads(out)
        out = self.w_concat(out)
        return out

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        d_head = d_model // self.n_head
        x = x.view(batch_size, seq_len, self.n_head, d_head)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size * self.n_head, seq_len, d_head)

    def concat_heads(self, x):
        batch_size, seq_len, d_head = x.size()
        d_model = d_head * self.n_head
        x = x.view(batch_size // self.n_head, self.n_head, seq_len, d_head)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size // self.n_head, seq_len, d_model)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(ffn_hidden, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x):
        _x = x
        x = self.self_attn(x, x, x)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, input_size, n_classes=6):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, 256),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.seq(x)
        return x

class Informer(nn.Module):
    def __init__(self, d_model, n_head, seq_len, ffn_hidden, n_layers, drop_prob, feature_dim):
        super(Informer, self).__init__()
        self.encoder_input_layer = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(max_seq_len=seq_len, d_model=d_model)
        self.encoder = Encoder(d_model, ffn_hidden, n_head, n_layers, drop_prob)
        self.class_head = ClassificationHead(256, n_classes=6)  # Adjusted input size


    def forward(self, src):
        src = self.encoder_input_layer(src)
        src = self.pos_encoder(src)
        enc_src = self.encoder(src)
        # Flatten the output of the encoder before passing it to the classification head
        enc_src_flat = enc_src.view(enc_src.size(0), -1)
        # print("Size of flattened encoder output:", enc_src_flat.size())  # Print the size of enc_src_flat
        cls_res = self.class_head(enc_src_flat)
        # print("Size of classification head output:", cls_res.size())  # Print the size of the output from the classification head
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

sequence_len = 40000
batch_size = 2
feature_dim = 40000
n_head = 2
n_layers = 1
drop_prob = 0.1
d_model = 128
ffn_hidden = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dataloaders = MyDataLoader(batch_size=batch_size).get_data_loader()
model = Informer(d_model=d_model, n_head=n_head, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layers, drop_prob=drop_prob, feature_dim=feature_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

model = train_model(dataloaders, model, optimizer, num_epochs=20)
torch.save(model.state_dict(), 'informer')
