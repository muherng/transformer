import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embed = nn.Linear(input_dim, d_model)  # Changed to a linear layer for real-valued input
        self.pos_encoder = self.create_pos_encoding(max_seq_length, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                        dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, 1)  # Output layer adjusted for regression

        self.init_weights()

    def create_pos_encoding(self, max_len, d_model):
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        pos_encoding = pos_encoding.unsqueeze(0)
        return nn.Parameter(pos_encoding, requires_grad=False)

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embed(src) * math.sqrt(self.d_model)
        src += self.pos_encoder[:, :src.size(1)]
        output = self.transformer_encoder(src)
        output = self.output_layer(output)
        return output

# Example parameters and model instantiation
input_dim = 1  # Assuming each token is a real-valued number
d_model = 512
nhead = 8
num_encoder_layers = 6
dim_feedforward = 2048
max_seq_length = 100

model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length).to(device)

# Example usage
#tokens = torch.randn(9,1)  # Batch size of 10, each with a single real-valued feature
#tokens = tokens.unsqueeze(0)  # Add a batch dimension
#output = model(tokens)

# Assuming real-valued targets for regression
#targets = torch.randn(output.size())  # Generating some random targets for demonstration
#print('tokens: ', tokens.size())
#print('output: ', output.size())
#print('targets" ', targets.size())

# Use Mean Squared Error Loss for regression
#criterion = nn.MSELoss()
#loss = criterion(output, targets)

#print(loss)

import torch.optim as optim

# Set up the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Learning rate is set to 0.001 as a starting point

def train(model, data_loader, epochs, learning_rate):
    model.train()  # Set the model to training mode
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()  # Clear the gradients
            data = data.unsqueeze(2)
            target = target.unsqueeze(2)
            #print('data: ', data.size())
            #print('target: ', target.size())
            output = model(data)  # Forward pass: compute the output
            loss = criterion(output, target)  # Compute the loss
            loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()  # Perform a single optimization step (parameter update)
            total_loss += loss.item()  # Accumulate the loss
            if epoch%100 == 1:
                print('data: ', data[0,:,:])
                print('output: ', output[0,:,:])
                print('target: ', target[0,:,:])
            
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
from torch.utils.data import Dataset, DataLoader

# Synthetic data generation
def generate_synthetic_data(num_sequences=1000, seq_length=3):
    """
    Generates synthetic data: a list of sequences of consecutive integers.
    
    Parameters:
    - num_sequences (int): Number of sequences to generate.
    - seq_length (int): Length of each sequence.
    
    Returns:
    - torch.Tensor: A tensor containing the generated sequences.
    """
    #data = torch.randn(num_sequences,2)
    data = [torch.arange(start=0, end=seq_length) for i in range(1, num_sequences+1)]
    return torch.stack(data).float().to(device)

class IntegerSequenceDataset(Dataset):
    """Dataset for sequences of consecutive integers."""
    
    def __init__(self, data):
        """
        Initializes the dataset with synthetic data.
        
        Parameters:
        - data (torch.Tensor): The synthetic data tensor.
        """
        self.data = data
    
    def __len__(self):
        """Returns the number of sequences in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves an input-target pair from the dataset at the specified index.
        
        Parameters:
        - idx (int): Index of the item.
        
        Returns:
        - tuple: (input_sequence, target_sequence) where target_sequence is the input_sequence shifted by one position.
        """
        input_sequence = self.data[idx][:-1]  # Input is the sequence except the last element
        target_sequence = self.data[idx][1:]  # Target is the sequence shifted by one position
        return input_sequence, target_sequence

    
# Example usage
num_sequences = 1000
seq_length = 10
synthetic_data = generate_synthetic_data(num_sequences, seq_length)
# Create the dataset
dataset = IntegerSequenceDataset(synthetic_data)

# Create a DataLoader
batch_size = 32
learning_rate = 0.001
epochs = 1000
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
train(model, data_loader, epochs, learning_rate)
   