import numpy as np
import torch, torch.nn as nn
from torch.autograd import Variable 
import sys
import torch.utils.data as Data
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#import os
#os.environ['CUDA_VISIBLE_DEVICES']='1, 2, 3'
glove_file = "./"
train_file = "./"
word2idx = {} 
vectors = []
test = []
train = []
len_max = 0
#with open("./KBP-SF48-master/train_sf.txt", "rb") as f:  
with open(train_file+"/train_sf.txt", "rb") as f:
    for l in f:
        line = l.decode().split("\t")
        tmp = len(line[1].split(" "))
        if  tmp > len_max:
            len_max = tmp 
        train.append(line[1])
        test.append(line[0])


unique = set()
#max = 0
#with open("./KBP-SF48-master/train_sf.txt", "rb") as f:    
with open("./train_sf.txt", "rb") as f:
    for l in f:
        line = l.decode().split("\t")
        chars = list(line[1])
        unique.update(chars)
        #if max< len(chars):
        #    max = len(chars)

unique = list(unique)
char2idx = {}
idx2char = {}

char2idx[""] = 0
idx2char[0] = ""
for i, char in enumerate(unique):
    char2idx[char] = i+1
    idx2char[i+1] = char
print(len(char2idx))


def char2idx_array(sentence_list, timestep, char2idx, length=9):

    idx_array = np.zeros((len(sentence_list), timestep,  length))
    for i, x in enumerate(sentence_list):
       
        idx = 0
        words = x.split(" ")
        #print(words)
        for idx, word in enumerate(words):
            idx_tmp = np.empty((0, 1), int)
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
#print(train[1])
#for char in x2[1][2]:
#    print(idx2char[char])
#sys.exit()

#print(x2)
#print(x2.shape)
#print(x2[1, 1, :])




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        #self.fc1 = nn.Linear(320, 50)
        #self.fc2 = nn.Linear(50, 10)

    def forward(self, x):

        embedded = self.embedding(x)
        batch, timestep, charlen, dim=embedded.shape
        #x = embedded.view(16, -1)
        embedded = embedded.view(batch * timestep, 1, charlen , dim)
        x = F.relu(F.max_pool2d(self.conv1(embedded), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #print(x.shape)
        #x = x.view(-1, 460)
        #sys.exit()
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        #return F.log_softmax(x, dim=1)
        return x



class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        #self.cnn = CNN()
        self.vocab_size = 83
        self.embedding_dim = 50
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.convolutions = []
        self.filter_num_width = [(25, 1), (50, 2), (75, 3), (100, 4), (125, 5), (150, 6)]
        
        for out_channel, filter_width in self.filter_num_width:
            self.convolutions.append(
                nn.Conv2d(
                    1,           # in_channel
                    out_channel, # out_channel
                    kernel_size=(50, filter_width), # (height, width)
                    bias=True
                    )
            )
        self.input_dim = sum([x for x, y in self.filter_num_width])
        self.batch_norm = nn.BatchNorm1d(self.input_dim, affine=False)
        self.num_classes = 41
        self.rnn = nn.GRU(
            input_size=525, 
            hidden_size=64, 
            num_layers=1,
            batch_first=True)
        #self.linear = nn.Linear(64,10)
        self.fc = nn.Linear(64, self.num_classes) 
        if True:
            for x in range(len(self.convolutions)):
                self.convolutions[x] = self.convolutions[x].cuda()
            self.rnn = self.rnn.cuda()
            self.embedding = self.embedding.cuda()
            self.batch_norm = self.batch_norm.cuda()    
    
    def conv_layers(self, x):
        chosen_list = list()
        for conv in self.convolutions:
            feature_map = F.tanh(conv(x))
            # (batch_size, out_channel, 1, max_word_len-width+1)
            chosen = torch.max(feature_map, 3)[0]
            # (batch_size, out_channel, 1)            
            chosen = chosen.squeeze()
            # (batch_size, out_channel)
            chosen_list.append(chosen)
        return torch.cat(chosen_list, 1)

    def forward(self, x):

        gru_batch_size = x.size()[0]
        gru_seq_len = x.size()[1]
        x = x.contiguous().view(-1, x.size()[2])
        # [num_seq*seq_len, max_word_len]
        #print(x.shape)
        x = self.embedding(x)

        # [num_seq*seq_len, max_word_len, char_emb_dim]
        #print(x.shape)
        #x = x.view(x.size()[0], 1, x.size()[1], -1)
        #print(x.shape)
        x = torch.transpose(x.view(x.size()[0], 1, x.size()[1], -1), 2, 3)
        #print(x.shape)
        # [num_seq*seq_len, 1, max_word_len, char_emb_dim]
        x = self.conv_layers(x)
        # [num_seq*seq_len, total_num_filters]
        x = self.batch_norm(x)
        #print(x)
        x = x.contiguous().view(gru_batch_size, gru_seq_len, -1)
        #print(x.shape)
        h0 = Variable(torch.rand(1, x.size(0), 64)).cuda()
        #c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        packed_h, packed_h_t = self.rnn(x, h0)
        decoded = packed_h_t[-1]
        #r_out, (h_n, h_c) = self.rnn(r_in)
        #r_out2 = self.linear(r_out[:, -1, :])
        logit = self.fc(decoded)
        #return decoded
        return F.log_softmax(logit, dim=1)




#dataset = Data.TensorDataset(torch.LongTensor(x), y, torch.LongTensor(x2))
y = torch.LongTensor(np.asarray(test, dtype=np.float32))
dataset = Data.TensorDataset(torch.LongTensor(x2), y)
train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=8,
                                           shuffle=True)

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


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

        self.connect = nn.Linear(128, 41)
    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear
        x = self.connect(x)
        return F.log_softmax(x, dim=1)


 
model2 = Combine()
#train(23)
#sys.exit()


if torch.cuda.is_available():
    #model1.cuda()
    model2.cuda()
    #model3.cuda()
    print("model will use GPU")



optimizer = optim.Adam(model2.parameters())


#model.train()

def train(model2, optimizer):
    
    epoch_loss = 0
    epoch_acc = 0
    
    iteration = 0    
    #model1.train()
    model2.train()
    #model3.train()
    for epoch in range(1, 100):
        for batch_idx, (data2, target) in enumerate(train_loader):
            #data, target, =data_helpers.sorting_sequence(data, target)

            if torch.cuda.is_available():
                target = Variable(target).cuda()
                data2 = Variable(data2).cuda()
            else:
                target = Variable(target)
                data2 = Variable(data2)

            #print(data.shape)
            #sys.exit()
        
            #optimizer.zero_grad()
            #first = model1(data)
            #print(first.shape)
            logit = model2(data2)
            #print(second.shape)
            #x = torch.cat((first, second), dim =1)
            #logit = model3(x)
            #loss = criterion(predictions, target)
            #print(target)
            #print(torch.max(target, 1)[1])
            #print(logit)
            loss = F.nll_loss(logit, target)
            #loss = nn.CrossEntropyLoss(logit, target)
            #print(loss)
    
            loss.backward()
        
            optimizer.step()


            iteration += 1

            #if args.iter % args.log_interval == 0:
            #print(torch.max(logit, 1)[1])
            #print(torch.max(logit, 1)[1].shape)
            #print(target)
            corrects_data = (torch.max(logit, 1)[1] == target).data
            #print(corrects_data)
            corrects = corrects_data.sum()
            #print(corrects)
            accuracy = 100.0 * corrects / len(target)
            print("Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})".format(iteration,
                                                                              loss.data[0],
                                                                              accuracy,
                                                                              corrects,
                                                                              len(target)))
            #if args.iter % args.dev_interval == 0:
            #    dev(model)

            #if args.iter % args.save_interval == 0:
                #if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                #torch.save(model, save_path)
        
            #epoch_loss += loss.item()
            #print(epoch_loss)
            #epoch_acc += acc.item()
        
        #return epoch_loss




train(model2, optimizer)







