import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        memory = torch.zeros_like(tgt)
        output = self.transformer_decoder(tgt, memory, tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.output_layer(output)
        return output

class TokenDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index:index+self.sequence_length], dtype=torch.long),
            torch.tensor(self.data[index+1:index+self.sequence_length+1], dtype=torch.long),
        )

def train(model, dataloader, epochs=10):
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print('epoch: ', epoch)
        total_loss = 0
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.view(-1, model.vocab_size), tgt.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")
        if epoch%10 == 0: 
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")
            print('input: ', src[0,:])
            print('target: ', tgt[0,:])
            val,indices = torch.max(output,2)
            print('output: ', indices[0,:])
#parameters
vocab_size = 5001
epochs=1000


#generate synthetic data
data_s = [1,1]
for i in range(2,1000):
    data_s.append((data_s[i-1] + data_s[i-2])%5000)
    
#print('data short: ', data_short)

#adding extra token
data = []
for i in range(int(len(data_s)/100)):
    data = data + data_s[100*i:100*i+100] + [5000] 
    
print('data: ', data)
# Example usage:
dataset = TokenDataset(data=data, sequence_length=100)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = TransformerDecoderModel(vocab_size).to(device)
train(model,dataloader,epochs=epochs)

#TODO: Expand number of tokens to 5000?
#memory keyword? 
#training format? 
