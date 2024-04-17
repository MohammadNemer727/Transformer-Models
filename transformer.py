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
from torch import nn,optim 
# from torchinfo import summary 
import numpy as np 
 
class Mydataset(Dataset):
    def __init__(self,x,y)  :
        super().__init__()
        #file_out = pd.read_csv(fileName)
        #x = file_out.iloc[:,:-1].values
        #y = file_out.iloc[:,-1:].values 
        self.X = torch.tensor(x)
        self.Y = torch.tensor(y)
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index): 
        return self.X[index].unsqueeze(1), self.Y[index] 
    

class myDataLoader():
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        # file_out_train = pd.read_csv('/content/drive/MyDrive/dataset/mitbih_train.csv') 

        # x_train = file_out_train.iloc[:,:-1].values
        # y_train = file_out_train.iloc[:,-1:].astype(dtype=int).values 
        # x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.15 )
        

        #print("x_train shape on batch size =  " + str(x_train.shape ))
        #print('x_val shape on batch size =  ' + str(x_val.shape))
        #print('y_train shape on batch size =  '+ str(y_train.shape ))
        #print('y_val shape on batch size =  ' + str( y_val.shape) )
        no_points = 1000
        train_data = np.empty((0, no_points*8), float)
        test_data = np.empty((0, no_points*8), float)
        train_label = []
        test_label = []

        folder_dir = './data/full'
        classes = os.listdir(folder_dir)
        train_counter = 0

        for class_name in classes:
            class_path = os.path.join(folder_dir, class_name)
            chosen_files = []
            k = 2 # Number of files to choose randomly from each class
            for _ in range(k):
                csv_file = random.choice(os.listdir(class_path))
                while csv_file in chosen_files:
                    csv_file = random.choice(os.listdir(class_path))
                chosen_files.append(csv_file)
                print("loading ",class_path," ",csv_file)
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
                            train_counter =0
                except Exception as e:
                    print(f"Error processing file {csv_file}: {e}")

        # Encoding labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(train_label)

        label_encoder = LabelEncoder()
        y_test_encoded = label_encoder.fit_transform(test_label)
        print("train data ",train_data.shape)
        print("train label ",y_train_encoded.shape)
        print("test data ",test_data.shape)
        print("test data ",y_test_encoded.shape)
        train_set= Mydataset(x= train_data, y= y_train_encoded) 

        val_set= Mydataset(x= test_data, y= y_test_encoded) 

        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True,  ),
            'val': DataLoader(val_set, batch_size=batch_size, shuffle=True,  )
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
        dropout: float=0.1, 
        max_seq_len: int=5000, 
        d_model: int=512,
        batch_first: bool=False    ): 
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
    def __init__(self,d_model, seq_len , details, n_classes: int = 6):
      super().__init__()
      self.norm = nn.LayerNorm(d_model)
      self.details = details
      #self.flatten = nn.Flatten()
      self.seq = nn.Sequential( nn.Flatten() , nn.Linear(d_model * seq_len , 512) ,nn.ReLU(),nn.Linear(512, 256)
                               ,nn.ReLU(),nn.Linear(256, 128),nn.ReLU(),nn.Linear(128, n_classes))
 
    def forward(self,x):

      if self.details:  print('in classification head : '+ str(x.size())) 
      x= self.norm(x)
      #x= self.flatten(x)
      x= self.seq(x)
      if self.details: print('in classification head after seq: '+ str(x.size())) 
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
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """
    def __init__(self, details):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.details = details
    def forward(self, q, k, v ,e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()
        
        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        
        if self.details: print('in Scale Dot Product, k_t size: '+ str(k_t.size()))
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product


        if self.details: print('in Scale Dot Product, score size: '+ str(score.size()))
        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        if self.details: print('in Scale Dot Product, score size after softmax : '+ str(score.size()))

        if self.details: print('in Scale Dot Product, v size: '+ str(v.size()))
        # 4. multiply with Value
        v = score @ v

        if self.details: print('in Scale Dot Product, v size after matmul: '+ str(v.size()))
        return v, score

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, details):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention( details=details)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.details = details

    def forward(self, q, k, v ):
        # 1. dot product with weight matrices

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        if self.details: print('in Multi Head Attention Q,K,V: '+ str(q.size()))
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        if self.details: print('in splitted Multi Head Attention Q,K,V: '+ str(q.size()))
        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v )
        
        if self.details: print('in Multi Head Attention, score value size: '+ str(out.size()))
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        if self.details: print('in Multi Head Attention, score value size after concat : '+ str(out.size()))
        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size() 
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob,details):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head, details=details)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.details = details
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x )
        
        if self.details: print('in encoder layer : '+ str(x.size()))
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        if self.details: print('in encoder after norm layer : '+ str(x.size()))
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        if self.details: print('in encoder after ffn : '+ str(x.size()))
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob,details, device):
        super().__init__()

        
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head
                                                  ,details=details,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x ): 
        for layer in self.layers:
            x = layer(x ) 
        return x


class Transformer(nn.Module):

    def __init__(self,device, d_model=100, n_head=4, max_len=5000, seq_len=200,
                 ffn_hidden=128, n_layers=2, drop_prob=0.1, details =False):
        super().__init__() 
        self.device = device
        self.details = details 
        self.encoder_input_layer = nn.Linear(   
            in_features=1, 
            out_features=d_model 
            )
   
        self.pos_emb = PostionalEncoding( max_seq_len=max_len,batch_first=False, d_model=d_model, dropout=0.1) 
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head, 
                               ffn_hidden=ffn_hidden, 
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               details=details,
                               device=device)
        self.classHead = ClassificationHead(seq_len=seq_len,d_model=d_model,details=details,n_classes=6)

    def forward(self, src ): 
        if self.details: print('before input layer: '+ str(src.size()) )
        src= self.encoder_input_layer(src)
        if self.details: print('after input layer: '+ str(src.size()) )
        src= self.pos_emb(src)
        if self.details: print('after pos_emb: '+ str(src.size()) )
        enc_src = self.encoder(src) 
        cls_res = self.classHead(enc_src)
        if self.details: print('after cls_res: '+ str(cls_res.size()) )
        return cls_res
    
def cross_entropy_loss(pred, target):

    criterion = nn.CrossEntropyLoss()
    #print('pred : '+ str(pred ) + ' target size: '+ str(target.size()) + 'target: '+ str(target )+   ' target2: '+ str(target))
    #print(  str(target.squeeze( -1)) )
    lossClass= criterion(pred, target ) 

    return lossClass


def calc_loss_and_score(pred, target, metrics): 
    softmax = nn.Softmax(dim=1)

    pred =  pred.squeeze( -1)
    target= target.squeeze( -1).long()

    ce_loss = cross_entropy_loss(pred, target)
    metrics['loss'] .append( ce_loss.item() )
    pred = softmax(pred )
    _,pred = torch.max(pred, dim=1)
    metrics['correct']  += torch.sum(pred ==target ).item()
    metrics['total']  += target.size(0) 
    return ce_loss
def print_metrics(main_metrics_train,main_metrics_val,metrics, phase):
    correct= metrics['correct']  
    total= metrics['total']  
    accuracy = 100*correct / total
    loss= metrics['loss'] 
    if(phase == 'train'):
        main_metrics_train['loss'].append( np.mean(loss)) 
        main_metrics_train['accuracy'].append( accuracy ) 
    else:
        main_metrics_val['loss'].append(np.mean(loss)) 
        main_metrics_val['accuracy'].append(accuracy ) 
    result = "phase: "+str(phase) \
    +  ' \nloss : {:4f}'.format(np.mean(loss))   +    ' accuracy : {:4f}'.format(accuracy)        +"\n"
    return result 
def train_model(dataloaders,model,optimizer, num_epochs=100): 
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_dict= dict()
    train_dict['loss']= list()
    train_dict['accuracy']= list() 
    val_dict= dict()
    val_dict['loss']= list()
    val_dict['accuracy']= list() 

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10) 

        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = dict()
            metrics['loss']=list()
            metrics['correct']=0
            metrics['total']=0
 
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.int)
                # zero the parameter gradients
                optimizer.zero_grad()


                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    #print('outputs size: '+ str(outputs.size()) )
                    loss = calc_loss_and_score(outputs, labels, metrics)   
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                #print('epoch samples: '+ str(epoch_samples)) 
            print(print_metrics(main_metrics_train=train_dict, main_metrics_val=val_dict,metrics=metrics,phase=phase ))
            epoch_loss = np.mean(metrics['loss'])
        
            if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss 

    print('Best val loss: {:4f}'.format(best_loss))

device = torch.device("cpu")
sequence_len=8000 # sequence length of time series
max_len=5000 # max time series sequence length 
n_head = 2 # number of attention head
n_layer = 1# number of encoder layer
drop_prob = 0.1
d_model = 2 # number of dimension ( for positional embedding)
ffn_hidden = 128 # size of hidden layer before classification 
feature = 1 # for univariate time series (1d), it must be adjusted for 1. 
batch_size = 4

model =  Transformer(d_model=d_model, n_head=n_head, max_len=max_len, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, details=False,device=device).to(device=device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
print("loading data")
dataloaders= myDataLoader(batch_size=batch_size).getDataLoader()
print("started training")
model_normal_ce = train_model(dataloaders=dataloaders, model=model, optimizer=optimizer, num_epochs=20)
torch.save(model.state_dict(), 'myModel')
