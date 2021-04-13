import numpy as np
import os
import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def one_hot(data, emsize):
    output = np.empty((len(data), len(data[0]), emsize))
    for i in range (len(data)):
        for j in range (len(data[0])):
            literal = int(data[i][j].item())
            row = [0, 0, 0, 0]
            if (literal < 4):
                row[literal] = 1
            else:
                row = [1, 1, 1, 1]
            output[i][j] = row
      
    output = torch.from_numpy(output)
    return output
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        #if self.src_mask is None or self.src_mask.size(0) != len(src):
        #    device = src.device
        #    mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #    self.src_mask = mask

        #src = one_hot(src, self.ninp)
        #src = src.cuda()
        src = self.encoder(src)*math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        src = src.to(torch.float32)
        src.requires_grad_(True)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        #output = src
        return output

ntokens = 1024*64 # the size of vocabulary
emsize = 128 # embedding dimension
nhid = 256 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4 # the number of heads in the multiheadattention models
dropout = 0 # the dropout value
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
#model.load_state_dict(torch.load("word5.pkl"))
criterion = nn.CrossEntropyLoss()
start_time= time.time()
data = torch.load("train_data.pt")
targets = torch.load("train_targets.pt")
batch_size = 64

file = open('log.txt', 'a')
bptt = 99
print (len(data[0])/batch_size)
lr = 0.001 # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print("Starting training")
for epoch in range(0,200):
    for i in range(int(len(data[0])/batch_size)):
        optimizer.zero_grad()
        input_batch = data[:, (i*batch_size):((i+1)*batch_size)].type(torch.long).to("cuda")
        target_batch = targets[i*bptt*batch_size:(i+1)*bptt*batch_size].type(torch.long).to("cuda")
        #input_batch_one_hot = one_hot(input_batch, emsize)
        output = model(input_batch)
        loss = criterion(output.view(-1, ntokens), target_batch)
        loss.backward()
        optimizer.step()
        if (i%200 == 0):
            print ("Epoch: %i, Loss: %5.5f, Time elaped: %8.1f" % (epoch, loss.item(), time.time() - start_time))
            file.write("Epoch: %i, Loss: %5.5f, Time elapsed: %8.1f\n" % (epoch, loss.item(), time.time() - start_time))	    
        
    torch.save(model.state_dict(), "LM.pkl")
file.close()
