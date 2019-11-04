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
        self.num_classes = 41
        self.rnn = nn.GRU(
            input_size=460, 
            hidden_size=64, 
            num_layers=1,
            batch_first=True)
        #self.linear = nn.Linear(64,10)
        self.fc = nn.Linear(64, self.num_classes) 
    def forward(self, x):
        batch_size, timesteps, charlens = x.size()
        #c_in = x.view(batch_size * timesteps, charlens)
        #for x in c_in:
        #    print(x)
        #print(c_in.shape)
        c_out = self.cnn(x)
        r_in = c_out.view(batch_size, timesteps, -1)
        h0 = Variable(torch.rand(1, r_in.size(0), 64)).cuda()
        #c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        packed_h, packed_h_t = self.rnn(r_in, h0)
        decoded = packed_h_t[-1]
        #r_out, (h_n, h_c) = self.rnn(r_in)
        #r_out2 = self.linear(r_out[:, -1, :])
        logit = self.fc(decoded)
        #return decoded
        return F.log_softmax(logit, dim=1)




dataset = Data.TensorDataset(torch.LongTensor(x), y, torch.LongTensor(x2))
#dataset = Data.TensorDataset(torch.LongTensor(x2), y)
train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=16,
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


 
model1 = RNN()
model2 = Combine()
model3 = Highway(128, 1, f= torch.nn.functional.relu)
#train(23)
#sys.exit()


if torch.cuda.is_available():
    #model1.cuda()
    model2.cuda()
    #model3.cuda()
    print("model will use GPU")



optimizer = optim.Adam(model3.parameters())


#model.train()

def train(model1, model2, model3, optimizer):
    
    epoch_loss = 0
    epoch_acc = 0
    
    iteration = 0    
    #model1.train()
    model2.train()
    #model3.train()
    for epoch in range(1, 100):
        for batch_idx, (data, target, data2) in enumerate(train_loader):
            #data, target, =data_helpers.sorting_sequence(data, target)

            if torch.cuda.is_available():
                data, target = Variable(data).cuda(), Variable(target).cuda()
                data2 = Variable(data2).cuda()
            else:
                data, target = Variable(data), Variable(target)
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
            loss = F.nll_loss(logit, target)
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




train(model1, model2, model3, optimizer)







