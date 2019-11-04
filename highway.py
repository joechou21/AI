import numpy as np
import torch, torch.nn as nn
from torch.autograd import Variable 
import sys
import torch.utils.data as Data
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

test = []
train = []
len_max = 0
with open("./KBP-SF48-master/train_sf.txt", "rb") as f:  
#with open(train_file+"/train_sf.txt", "rb") as f:
    for l in f:
        line = l.decode().split("\t")
        tmp = len(line[1].split(" "))
        if  tmp > len_max:
            len_max = tmp 
        train.append(line[1])
        test.append(line[0])

def word2idx_array(sentence_list, length, word2idx):

    idx_array = np.zeros((len(sentence_list), length))
    len_max = 0
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


x, word2idx = word2idx_array(train, len_max, word2idx)
y = torch.LongTensor(np.asarray(test, dtype=np.float32))

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
        #c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        packed_h, packed_h_t = self.rnn(x, h0)
        decoded = packed_h_t[-1]
        # Decode hidden state of last time step
        #logit = self.fc(decoded)
        #return F.log_softmax(logit, dim =1)
        print(decoded.shape)
        return decoded


unique = set()
#max = 0
with open("./KBP-SF48-master/train_sf.txt", "rb") as f:    
#with open("./train_sf.txt", "rb") as f:
    for l in f:
        line = l.decode().split("\t")
        chars = list(line[1])
        unique.update(chars)
        #if max< len(chars):
        #    max = len(chars)

unique = list(unique)
char2idx = {}


for i, char in enumerate(unique):
    char2idx[char] = i

print(len(char2idx))


def char2idx_array(sentence_list, timestep, char2idx, length=9):

    idx_array = np.zeros((len(sentence_list), timestep,  length))
    for i, x in enumerate(sentence_list):
        idx_tmp = np.empty((0, 1), int)
        idx = 0
        words = x.split()
        for idx, word in enumerate(words):
            for char in list(word):
                idx_tmp = np.vstack((idx_tmp, int(char2idx[char])))
        
            if length > len(idx_tmp):
                num_zeros = length - len(idx_tmp)
                zeros_array = np.zeros((1, num_zeros))
                sen_max_len = np.hstack((idx_tmp.T, zeros_array))
            #print(idx)
                idx_array[i][idx] = sen_max_len
            else:
            #print(idx)
                idx_array[i][idx] = idx_tmp[:length].T

    return idx_array



x2 = char2idx_array(train, len_max, char2idx, 9)

#print(x2)
#print(x2.shape)
#print(x2[1, 1, :])




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.vocab_size = 82
        self.embedding_dim = 50
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        #self.fc1 = nn.Linear(320, 50)
        #self.fc2 = nn.Linear(50, 10)

    def forward(self, x):

        embedded = self.embedding(x)
        batch, timestep, charlen, dim=embedded.shape
        #x = embedded.view(16, -1)
        embedded = embedded.view(batch * timestep, 1, charlen , dim)
        #print("here")
        x = F.relu(F.max_pool2d(self.conv1(embedded), 2))
        #print("pass")
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        #print(x.shape)
        x = x.view(-1, 460)

        #embedded = embedded.permute(1, 0, 2)
        
        #embedded = [batch size, sent len, emb dim]
        
        #pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        
        #pooled = [batch size, embedding_dim]
                
        #logit = self.fc(pooled)
        #return F.log_softmax(logit)



        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        #return F.log_softmax(x, dim=1)
        return x



class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.GRU(
            input_size=460, 
            hidden_size=64, 
            num_layers=1,
            batch_first=True)
        self.linear = nn.Linear(64,10)

    def forward(self, x):
        batch_size, timesteps, charlens = x.size()
        #c_in = x.view(batch_size * timesteps, charlens)
        #for x in c_in:
        #    print(x)
        #print(c_in.shape)
        c_out = self.cnn(x)
        r_in = c_out.view(batch_size, timesteps, -1)
        h0 = Variable(torch.rand(1, x.size(0), 64))
        #c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        packed_h, packed_h_t = self.rnn(r_in, h0)
        decoded = packed_h_t[-1]
        #r_out, (h_n, h_c) = self.rnn(r_in)
        #r_out2 = self.linear(r_out[:, -1, :])
        print(decoded.shape)
        return decoded
        #return F.log_softmax(r_out2, dim=1)




dataset = Data.TensorDataset(torch.LongTensor(x), y, torch.LongTensor(x2))
#dataset = Data.TensorDataset(torch.LongTensor(x2), y)
train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=16,
                                           shuffle=False)

#model = Combine()
import torch.optim as optim

#if args.cuda:
#model.cuda()
#optimizer = optim.Adam(model.parameters())

#def train(epoch):
    #mode2.train()
    #model.train()
    #for batch_idx, (data, target) in enumerate(train_loader):
        

        #data = torch.LongTensor(data)
        #if args.cuda:
        #    data, target = data.cuda(), target.cuda()
            

        
        #data, target = Variable(data), Variable(target)
        #optimizer.zero_grad()
        #print(data.shape)
        #output = model(data)

        
        #loss = F.nll_loss(output, target)
        #loss.backward()
        #optimizer.step()

 
model1 = RNN()
model2 = Combine()

#train(23)
#sys.exit()

if torch.cuda.is_available():
    model1.cuda()
    model2.cuda()
    print("model will use GPU")




#class HighWay(nn.Module):
#    def __init_(self, modelA, modelB):
#        super(MyEnsemble, self).__init__()
#        self.modelA = modelA
#        self.modelB = modelB
#        self.output_dim = 41
#        self.classifier = nn.Linear(4,2)
#    def forward(self, x1, x2):
#        x1 = self.modelA(x1)
#        x2 = self.modelB(x2)
#        x = torch.cat((x1,x2), dim =1)
#        x = self.classifier(F.relu(x))

 #       return x


#model3 = HighWay(model1, model2)
#model3.cuda()





optimizer = optim.Adam(model1.parameters())


def train(optimizer):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model1.train()
    model2.train()
    
    for batch_idx, (data, target, data2) in enumerate(train_loader):
        #data, target, =data_helpers.sorting_sequence(data, target)

        if torch.cuda.is_available():
            data, target = Variable(data).cuda(), Variable(target).cuda()
            data2, target2 = Variable(data2).cuda(), Variable(target2).cuda()
        else:
            data, target = Variable(data).cuda(), Variable(target).cuda()
            data2, target2 = Variable(data2).cuda(), Variable(target2).cuda()

        #print(data.shape)
        #sys.exit()
        
        optimizer.zero_grad()
        first = model1(data)
        print(first.shape)
        second = model2(data2)
        print(second.shape)
        sys.exit()
        #predictions = model(data).squeeze(1)
        logit = mode3(first, second)
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




train(optimizer)



