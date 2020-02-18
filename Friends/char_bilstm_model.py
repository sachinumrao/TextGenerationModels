import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# Text Pre-processing
with open("../../..//Data/LM/friends_script_edit.txt", "r") as datafile:
    text = datafile.read()

# Convert text to lowercase
text = text.lower()

# Converisons
conversions = {
    '[' : '(',
    ']' : ')',
    '{' : ')',
    '}' : ')',
    '<' : '(',
    '>' : ')',
    '\n': ' '
}

for j in conversions.keys():
    text = text.replace(j, conversions[j])

# Remove unwanted chars
charset1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
            'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
            's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

charset2 = ['(', ')', '-', ':', '"', "'", '.', ',', '?', '!', ' ']

allowed_charset = charset1 + charset2

rtext = ''
for j in text:
    if j in allowed_charset:
        rtext += j

final_charset = sorted(list(set(rtext)))
corpus_length = len(rtext)
print("\nTotal Characters: ", len(final_charset))
print("\nCorpus Length: ", corpus_length)

# Build charatcer and id mappings
id2char = dict(enumerate(final_charset))
char2id = {value:key for key,value in id2char.items()}

def char2onehot(ch):
    out = np.zeros((len(final_charset), 1)).reshape(1,-1)
    out[0, char2id[ch]] = 1
    return out

# Model params
num_epochs = 5
seq_len = 32
step_size = 4
batch_size = 64
input_size = len(final_charset)
hidden_size = 128
n_layers = 2
dense1 = 512
dense2 = 128
dense3 = len(final_charset)
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build training set
sentences = []
labels = []

for i in range(0, corpus_length, step_size):
    substr = rtext[i : i+seq_len+1]
    if len(substr) == seq_len+1:
        sentences.append(substr[:-1])
        labels.append(substr[-1])
    else:
        pass

print("\nTraining Sentences: ", len(sentences))
print("\nTraining Labels: ", len(labels))

N = len(sentences)
num_batches = N//batch_size

def get_x(sents, batch_size):
    out = np.zeros((batch_size, seq_len, input_size))
    for i in range(batch_size):
        sent = sents[i]
        for j, ch in enumerate(sent):
            out[i, j, :] = char2onehot(ch)

    return out

def get_y(lbls, batch_size):
    out = np.zeros((batch_size))
    for i,ch in enumerate(lbls):
        out[i] = char2id[ch]
    
    return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define model architecture
class CharRNNModel(nn.Module):
    def __init__(self,
                batch_size,
                input_size,
                hidden_size,
                n_layers,
                dense1,
                dense2,
                dense3,
                bilstm=True):

        super(CharRNNModel, self).__init__()

        self.batch_size = batch_size
        self.hiddne_size = hidden_size
        self.bilstm = bilstm
        self.n_layers = n_layers
        self.lstm_layer = nn.LSTM(input_size,
                                    self.hiddne_size,
                                    num_layers = n_layers,
                                    batch_first=True,
                                    bidirectional=self.bilstm)

        self.dense_layer1 = nn.Linear(self.hiddne_size*2, dense1)
        self.dense_layer2 = nn.Linear(dense1, dense2)
        self.dense_layer3 = nn.Linear(dense2, dense3)

    def forward(self, x, prev_state):
        out, state = self.lstm_layer(x, prev_state)
        # reformat lstm output
        out = out[:, -1, :]
        out = F.relu(self.dense_layer1(out))
        out = F.relu(self.dense_layer2(out))
        out = self.dense_layer3(out)

        return out, state

    def zero_state(self, batch_size):
        if self.bilstm:
            return (torch.zeros(2*self.n_layers, batch_size, self.hiddne_size),
                    torch.zeros(2*self.n_layers, batch_size, self.hiddne_size))
        else:
            return (torch.zeros(1*self.n_layers, batch_size, self.hiddne_size),
                    torch.zeros(1*self.n_layers, batch_size, self.hiddne_size))

# Instantiate model
model = CharRNNModel(batch_size, input_size, hidden_size, n_layers, dense1, dense2, dense3)
model = model.float()
model = model.to(device)

print()
print(model)
print("\nModel Parameters: ", count_parameters(model))

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Settingup the training loop
prev_state = None
# Get zero state
prev_state = model.zero_state(batch_size)
h0 = prev_state[0].detach()
c0 = prev_state[1].detach()

for epoch in range(num_epochs):

    for i in range(num_batches):
        
        sents = sentences[i*batch_size : (i+1)*batch_size]
        lbls = labels[i*batch_size : (i+1)*batch_size]

        x = get_x(sents, batch_size)
        y = get_y(lbls, batch_size)

        x_train = torch.from_numpy(x).float()
        y_train = torch.LongTensor(y)

        x_train = x_train.to(device)
        y_train = y_train.to(device)

        # Clear previous gradients
        optim.zero_grad()

        # Forward pass
        out, state = model(x_train, (h0,c0))
        h0 = state[0].detach()
        c0 = state[1].detach()

        # Calculate loss
        loss = criterion(out, y_train)

        # Calculate gradients
        loss.backward()

        # Apply gradients in backprop
        optim.step()

        #Print progress
        if (i+1) % 100 == 0:
            print(f"""Epoch: {epoch+1}/{num_epochs}, Step: {i+1}/{num_batches}, Loss: {loss.item()}""")

# Add prediction loop
print("\nStarting Text generation...")
r1 = np.random.randint(0, corpus_length-seq_len)
intial_string = rtext[r1 : r1+seq_len]
generate_length = 30
test_batch_size = 1

model.eval()
with torch.no_grad():
    st = model.zero_state(test_batch_size)
    h1 = st[0]
    c1 = st[1]
    starter = intial_string
    for i in range(generate_length):
        x = get_x(starter, test_batch_size)
        x_train = torch.from_numpy(x).float()
        x_train = x_train.to(device)
        # print("x_train shape: ", x_train.shape)
        out, state = model(x_train, (h1,c1))
        # print("output shape: ", out.shape)
        h1 = state[0].detach()
        c1 = state[1].detach()
        p = F.softmax(out, dim=1).data
        m = Categorical(p)
        ch_id = m.sample().numpy()
        
        # Modify starter
        starter = starter[1:]
        # print(ch_id)
        # print(id2char)
        starter += id2char[ch_id[0]]
        intial_string += id2char[ch_id[0]]

print("\nOutput: ", intial_string)















