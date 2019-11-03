import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import torch
from torch.autograd import Variable 
import torch.utils.data as Data

y = []
train = []
unique = set()
max = 0
with open("./KBP-SF48-master/train_sf.txt", "rb") as f:
    
    for l in f:
        line = l.decode().split("\t")
        chars = list(line[1])
        unique.update(chars)
        if max< len(chars):
            max = len(chars)
        train.append(chars)
        y.append(line[0])



unique = list(unique)
char2idx = {}
for i, char in enumerate(unique):
    char2idx[char] = i


def char2idx_array(sentence_list, length ,char2idx):

    idx_array = np.zeros((len(sentence_list), length))
    for i, x in enumerate(sentence_list):
        idx_tmp = np.empty((0, 1), int)
        for char in x:  
            idx_tmp = np.vstack((idx_tmp, int(char2idx[char])))
        
        if length > len(idx_tmp):
            num_zeros = length - len(idx_tmp)
            zeros_array = np.zeros((1, num_zeros))
            sen_max_len = np.hstack((idx_tmp.T, zeros_array))
            idx_array[i] = sen_max_len
        else:
            idx_array[i] = idx_tmp[:length].T

    return idx_array



x = char2idx_array(train, max, char2idx)
y = torch.LongTensor(np.asarray(y, dtype=np.float32))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = 72
        self.embedding_dim = 50
        self.output_dim = 41
        self.pad_idx = 0
        #self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.fc = nn.Linear(self.embedding_dim, self.output_dim)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        embedded = self.embedding(text)
                
        #embedded = [sent len, batch size, emb dim]
        
        embedded = embedded.permute(1, 0, 2)
        
        #embedded = [batch size, sent len, emb dim]
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        
        #pooled = [batch size, embedding_dim]
                
        logit = self.fc(pooled)
        return F.log_softmax(logit)



dataset = Data.TensorDataset(torch.LongTensor(x), y)
train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=16,
                                           shuffle=False)


#criterion = nn.BCEWithLogitsLoss().cuda()
model = CNN()
if torch.cuda.is_available():
    model.cuda()
    print("model will use GPU")

import torch.optim as optim

optimizer = optim.Adam(model.parameters())


def train(model, optimizer):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target, =data_helpers.sorting_sequence(data, target)
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        
        predictions = model(data).squeeze(1)
        
        #loss = criterion(predictions, target)
        loss = F.nll_loss(logit, torch.max(target, 1)[1])
        print(loss)
    
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)






train(model, optimizer)



