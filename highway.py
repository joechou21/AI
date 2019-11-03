import numpy as np
import torch, torch.nn as nn
from torch.autograd import Variable 
import sys
import torch.utils.data as Data
glove_file = "./"
train_file = "./"
word2idx = {} 
vectors = []

with open(glove_file+"/glove.6B.50d.txt", "rb") as f:
    index = 1
    for l in f:
        line = l.decode().split()
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
        word2idx[line[0]] = index
        index+=1

y = []
train = []
with open(train_file+"/train_sf.txt", "rb") as f:
    for l in f:
        line = l.decode().split("\t")
        train.append(line[1])
        y.append(line[0])

def word2idx_array(sentence_list, length, word2idx):

    idx_array = np.zeros((len(sentence_list), length))
    count = 0
    for i, x in enumerate(sentence_list):
        idx_tmp = np.empty((0, 1), int)
        word_list = x.split(" ")
        
        for word in word_list:
            if word not in word2idx:
                word2idx[word] = len(word2idx) + 1  
            idx_tmp = np.vstack((idx_tmp, int(word2idx[word])))
        
        if length > len(idx_tmp):
            num_zeros = length - len(idx_tmp)
            zeros_array = np.zeros((1, num_zeros))
            sen_max_len = np.hstack((idx_tmp.T, zeros_array))
            idx_array[i] = sen_max_len
        else:
            idx_array[i] = idx_tmp[:length].T

    return idx_array, word2idx


x, word2idx = word2idx_array(train, 82, word2idx)
y = torch.LongTensor(np.asarray(y, dtype=np.float32))

weights_matrix = np.zeros((len(word2idx)+1, 50))
#word idx
for word in word2idx:
    try: 
        weights_matrix[word2idx[word]] = vectors[word2idx[word]]
    except:
        weights_matrix[word2idx[word]] = np.random.normal(scale=0.6, size=(50, ))


def create_emb_layer(matrix, non_trainable=False):
    matrix = np.asarray(matrix)
    num_embeddings, embedding_dim = matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.from_numpy(matrix)})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

emb, n, d= create_emb_layer(weights_matrix)
print(emb)


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.hidden_size = 512
        self.num_layers = 1
        self.num_classes = 41
        self.seq_len = 82
        self.embedding_dim = d
        self.embedding_num = n
        #self.embed = nn.Embedding(self.embedding_num + 1, self.embedding_dim)
        self.embed = emb
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, batch_first=True,
                              bidirectional= True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes) 
        # bidirection 
        self.num_layers *= 2

    def forward(self, input_x, lens = None):
        # Set initial states
        x = self.embed(input_x)  # dim: (batch_size, max_seq_len, embedding_size)
        #print(x.shape)
        #leng = torch.as_tensor(82.cpu(), dtype=torch.int64)
        #lens = torch.tensor(lens, dtype=torch.int64, device=torch.device('cpu'))
        #data, target, seq = data_helpers.sorting_sequence(data, target, seq, args)
        #packed_x = pack_padded_sequence(x, lengths = Variable(lens).cuda(), batch_first=True)

        h0 = Variable(torch.rand(self.num_layers, x.size(0), self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        packed_h, packed_h_t = self.rnn(x, h0)
        decoded = packed_h_t[-1]
        # Decode hidden state of last time step
        logit = self.fc(decoded)
        #return F.log_softmax(logit)
        return logit

model = RNN()


y = []
train = []
unique = set()
max = 0
#with open("./KBP-SF48-master/train_sf.txt", "rb") as f:    
with open("./train_sf.txt", "rb") as f:
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
        #print(text.shape)
        embedded = self.embedding(text)
        #print(embedded.shape)        
        #embedded = [sent len, batch size, emb dim]
        
        #embedded = embedded.permute(1, 0, 2)
        #print(embedded.shape)
        #embedded = [batch size, sent len, emb dim]
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        #print(pooled.shape)
        #pooled = [batch size, embedding_dim]
                
        logit = self.fc(pooled)
        
        return logit
        #return F.log_softmax(logit,dim=1)



dataset = Data.TensorDataset(torch.LongTensor(x), y)
train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=16,
                                           shuffle=False)


#criterion = nn.BCEWithLogitsLoss().cuda()
model = CNN()
if torch.cuda.is_available():
    model.cuda()
    print("model will use GPU")


class HighWay(nn.Module):
    def __init_(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(4,2)
    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1,x2), dim =1)
        x = self.classifier(F.relu(x))

        return x



import torch.optim as optim

optimizer = optim.Adam(model.parameters())


def train(model, optimizer):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target, =data_helpers.sorting_sequence(data, target)
        data, target = Variable(data).cuda(), Variable(target).cuda()
        #print(data.shape)
        #sys.exit()
        optimizer.zero_grad()
        
        #predictions = model(data).squeeze(1)
        logit = model(data)
        #loss = criterion(predictions, target)
        #print(target)
        #print(torch.max(target, 1)[1])
        loss = F.nll_loss(logit, target)
        #print(loss)
    
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        print(epoch_loss)
        #epoch_acc += acc.item()
        
    return epoch_loss




train(model, optimizer)



