import torch
import torch.nn as nn
import config

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm_size = config.lstm_size 
        self.embedding_dim = config.embed_dim 
        self.num_layers = config.n_layers 
        self.vocab_size = config.n_vocab 
        self.max_len = config.max_len 
        self.lstm_dropout = config.lstm_dropout 
        self.fc_dropout = config.fc_dropout 

        # Create Embedding Layer
        self.embedding = nn.Embedding(
                            num_embeddings=self.vocab_size,
                            embedding_dim=self.embedding_dim
        )

        # Create LSTM Layer
        self.lstm = nn.LSTM(
                        input_size=self.lstm_size,
                        hidden_size=self.lstm_size,
                        num_layers = self.num_layers,
                        dropout=self.lstm_dropout
        )

        # Create Output Layer
        self.fc = nn.Linear(self.lstm_size, self.vocab_size)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        out, state = self.lstm(embed, prev_state)
        logits = self.fc(out)
        return logits, state

    def init_state(self):
        base_state = (torch.zeros(self.num_layers, self.max_len, self.lstm_size),
                        torch.zeros(self.num_layers, self.max_len, self.lstm_size))

        return base_state