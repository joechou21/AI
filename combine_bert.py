import numpy as np
import torch, torch.nn as nn
from torch.autograd import Variable 
import sys
import torch.utils.data as Data
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from pytorch_pretrained_bert import BertTokenizer, BertModel
glove_file = "./"
train_file = "./" 
vectors = []
test = []
train = []
#with open("./KBP-SF48-master/train_sf.txt", "rb") as f:  
len_max = 0
with open(train_file+"/train_sf.txt", "rb") as f:
    for l in f:
        line = l.decode().split("\t")
        tmp = len(line[1].split(" "))
        if  tmp > len_max:
            len_max = tmp 
        train.append(line[1])
        test.append(line[0])

batch_size = 8

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train = [tokenizer.tokenize('[CLS] ' + sent + ' [SEP]') for sent in train] # Appending [CLS] and [SEP] tokens - this probably can be done in a cleaner way
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model = bert_model.cuda()

X_train_tokens = [tokenizer.convert_tokens_to_ids(sent) for sent in X_train]


for i, x in enumerate(X_train_tokens):
    #if len(x) > len_max:
    #    len_max = len(x)
    # 117 is the longest len of x in X_train_tokens
    cnt = 117- len(x)
    while cnt > 0:
        x.append(0)
        X_train_tokens[i]= x
        cnt-=1

x = torch.zeros((len(X_train_tokens), bert_model.config.hidden_size)).long()
with torch.no_grad():
    for stidx in range(0, len(X_train_tokens), batch_size):
        X = X_train_tokens[stidx:stidx + batch_size]
        X = torch.LongTensor(X).cuda()
        _, pooled_output = bert_model(X)
        x[stidx:stidx + batch_size,:] = pooled_output.cpu()

y = torch.LongTensor(np.asarray(test, dtype=np.float32))

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.hidden_size = 64
        self.num_layers = 1
        self.num_classes = 41
        self.embedding_dim = 50
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, batch_first=True,
                              bidirectional= True)
        #self.fc = nn.Linear(self.hidden_size, self.num_classes) 
        # bidirection 
        self.num_layers *= 2

    def forward(self, input_x, hidden):
        # Set initial states
        #x = self.embed(input_x)  # dim: (batch_size, max_seq_len, embedding_size
        print(input_x.shape)
        packed_h, packed_h_t = self.rnn(input_x, hidden)
        decoded = packed_h_t[-1]
        # Decode hidden state of last time step
        #logit = self.fc(decoded)
        #return F.log_softmax(logit, dim =1)
        return decoded


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
#print(len(char2idx))


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

class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        #self.cnn = CNN()
        self.modelA = RNN()
        self.vocab_size = 83
        self.embedding_dim = 50
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.convolutions = []
        self.highway_input_dim = 128
        #self.filter_num_width = [(25, 1), (50, 2), (75, 3), (100, 4), (125, 5), (150, 6)]
        self.filter_num_width = [(25, 1), (50, 2)]
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
        #self.batch_norm = nn.BatchNorm1d(self.input_dim, affine=False)
        self.num_classes = 41
        self.rnn = nn.GRU(
            input_size=75, 
            hidden_size=64, 
            num_layers=1,
            batch_first=True)
        #self.linear = nn.Linear(64,10)
        self.highway = Highway(self.highway_input_dim)
        self.fc = nn.Linear(self.highway_input_dim, self.num_classes) 
     
        if True:
            for x in range(len(self.convolutions)):
                self.convolutions[x] = self.convolutions[x].cuda()
            self.rnn = self.rnn.cuda()
            self.embedding = self.embedding.cuda()
            self.highway = self.highway.cuda()
            self.modelA = self.modelA.cuda()

    def conv_layers(self, x):
        chosen_list = list()
        for conv in self.convolutions:
            feature_map = torch.tanh(conv(x))
            # (batch_size, out_channel, 1, max_word_len-width+1)
            chosen = torch.max(feature_map, 3)[0]
            # (batch_size, out_channel, 1)            
            chosen = chosen.squeeze()
            # (batch_size, out_channel)
            chosen_list.append(chosen)
        return torch.cat(chosen_list, 1)

    def forward(self, x, x1, hidden, hidden1):
        x1 = self.modelA(x1, hidden1)
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
        #print(x.shape)
        # [num_seq*seq_len, total_num_filters]
        #x = self.batch_norm(x)
        #print(x)
        x = x.contiguous().view(gru_batch_size, gru_seq_len, -1)
        #print(x.shape)
        #h0 = Variable(torch.rand(1, x.size(0), 64)).cuda()
        #c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        packed_h, packed_h_t = self.rnn(x, hidden)
        x2 = packed_h_t[-1]
        input_highway = torch.cat((x1, x2), dim =1)
        result = self.highway(input_highway)
        logit = self.fc(result)
        #return decoded
        return F.log_softmax(logit, dim=1)





dataset = Data.TensorDataset(x, y, torch.LongTensor(x2))
print(len(dataset))
train_size = int(0.6 * len(dataset))
print(train_size)
val_size = (len(dataset) - train_size)//2
print(val_size)
test_size = val_size+1
print(test_size)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
#dataset = Data.TensorDataset(torch.LongTensor(x2), y)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=8,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=8,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=8,
                                           shuffle=False)


import torch.optim as optim



class Highway(nn.Module):
    """Highway network"""
    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, input_size, bias=True)

    def forward(self, x):
        t = F.sigmoid(self.fc1(x))
        return torch.mul(t, F.relu(self.fc2(x))) + torch.mul(1-t, x)        


model2 = Combine()
#model3 = Highway(128, 1, f= torch.nn.functional.relu)


if torch.cuda.is_available():
    #model1.cuda()
    model2.cuda()
    #model3.cuda()
    print("model will use GPU")



optimizer = optim.Adam(model2.parameters())
#optimizer = optim.Adam(list(model1.parameters()) + list(model2.parameters())
#    + list(model3.parameters()))

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def save_checkpoint(model, state, filename):

    model_is_cuda = next(model.parameters()).is_cuda
    model = model.module if model_is_cuda else model
    state['state_dict'] = model.state_dict()
    torch.save(state,filename)

def fscore(y_pred, y_true):

    #m = MultiLabelBinarizer().fit([y_true])
    #y_true = m.transform([y_true])
    #y_pred = m.transform(y_pred)
    micro = f1_score(y_true, y_pred, average='micro')
    macro = f1_score(y_true, y_pred, average='macro') 
    print(micro)
    print(macro)

    return micro, macro  
    #print(classification_report(y_true, y_pred))
    #sys.exit()
def eval(model2, file):
    
    model2.eval()
    corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0
    predicates_all, target_all = [], []
    hidden2 = Variable(torch.zeros(1, 8, 64).cuda())
    hidden = Variable(torch.zeros(2, 8, 64).cuda())
    with torch.no_grad():
        for batch_idx, (data, target, data2) in enumerate(val_loader):
            model2.zero_grad()

            size += len(target)
            if data2.size(0)!=8:
                continue
            if torch.cuda.is_available():
                data, target = Variable(data).cuda(), Variable(target).cuda()
                data2 = Variable(data2).cuda()
            else:
                data, target = Variable(data), Variable(target)
                data2 = Variable(data2)
            hidden2 = repackage_hidden(hidden2)
            hidden = repackage_hidden(hidden)
            logit = model2(data2, data, hidden2, hidden)
         
            predicates = torch.max(logit, 1)[1].view(target.size()).data
            accumulated_loss += F.nll_loss(logit, target, size_average = False).data
            corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            predicates_all += predicates.cpu().numpy().tolist()
            target_all += target.data.cpu().numpy().tolist()
    
    avg_loss = accumulated_loss / size
    accuracy = 100.0 * corrects / size
    model2.train()
    print('\nEvaluation - loss: {:.6f}  lr: {:.5f}  acc: {:.3f}% ({}/{}) '.format(avg_loss, 
                                                                       optimizer.state_dict()['param_groups'][0]['lr'],
                                                                       accuracy,
                                                                       corrects, 
                                                                       size))
    micro, macro = fscore(predicates_all, target_all)
    file.write(micro+" "+macro)
    print('\n')
    
    return avg_loss, accuracy





def train(model2, optimizer):
    
    iteration = 0    
    #model1.train()
    #model2.train()
    model2.train()
    hidden2 = Variable(torch.zeros(1, 8, 64).cuda())
    hidden = Variable(torch.zeros(2, 8, 64).cuda())

    best_acc = None
    file = open("./combine_bert.txt","a") 
    for epoch in range(1, 100):
        accuracy = 0
        for batch_idx, (data, target, data2) in enumerate(train_loader):
            #data, target, =data_helpers.sorting_sequence(data, target)
            if data2.size(0)!=8:
                continue
            if torch.cuda.is_available():
                data, target = Variable(data).cuda(), Variable(target).cuda()
                data2 = Variable(data2).cuda()
            else:
                data, target = Variable(data), Variable(target)
                data2 = Variable(data2)
            hidden2 = repackage_hidden(hidden2)
            hidden = repackage_hidden(hidden)
            #print(data.shape)
            #sys.exit()
        
            optimizer.zero_grad()
            logit = model2(data2, data, hidden2, hidden)
            #print(second.shape)
            #x = torch.cat((logit1, logit2), dim =1)
            #logit = model3(x)
            
            loss = F.nll_loss(logit, target)
            #
            #print(loss)
    
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(list(model1.parameters()) + list(model2.parameters())
            #    + list(model3.parameters()), 0.5)        
            optimizer.step()


            iteration += 1
            
            if iteration % 100 == 0:
                corrects_data = (torch.max(logit, 1)[1] == target).data
                corrects = corrects_data.sum()
                accuracy = 100.0 * corrects / len(target)
                #print("Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})".format(iteration,
                #                                                              loss.data[0],
                #                                                              accuracy,
                #                                                              corrects,
                #                                                              len(target)))
        # validation
        val_loss, val_acc = eval(model2, file)
        file.write("\t"+val_loss+"\t"+val_acc+"\t"+epoch+"\n")
        # save best validation epoch model
        if best_acc is None or val_acc > best_acc:
            file_path = '%s/AI_best.pth.tar' % ("./")
            print("\r=> found better validated model, saving to %s" % file_path)
            print(accuracy)
            print(val_acc)
            print(epoch)
                #save_checkpoint(model, 
                #            {'epoch': epoch,
                #            'optimizer' : optimizer.state_dict(), 
                #            'best_acc': best_acc},
                #            file_path)
            best_acc = val_acc
    

train(model2, optimizer)







