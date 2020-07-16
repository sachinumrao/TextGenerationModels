import enum
import numpy as np
import pickle
import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from friends_data import FriendsDataset
from model import LSTMModel
import config 
import time
import matplotlib.pyplot as plt

def train(model, train_dataset, test_dataset):
    batch_size = config.batch_size
    lr = config.learning_rate
    num_epochs = config.epochs 

    dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    test_loss = []
    epoch_test_loss = []
    epoch_train_loss = []
    

    for epoch in range(num_epochs):
        t_init = time.time()
        model.train()
        train_loss = []
        state_h, state_c = model.init_state()

        for batch_id, data in enumerate(dataloader):
            optimizer.zero_grad()
            x, y = data
            y_pred, state_h, state_c = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            print("Epoch:",epoch, ' Batch:',batch_id, 'Loss:',loss.item())

        # Perform validation 
        model.eval()
        state_h, state_c = model.init_state()
        for test_batch_id, test_data in enumerate(test_dataloader):
            x, y = test_data
            y_pred, state_h, state_c = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()
            test_loss.append(loss.item())

        epoch_train_loss.append(np.mean(train_loss))
        epoch_test_loss.append(np.mean(test_loss))
        t_finish = time.time()

        print("Time Taken: ", t_finish - t_init)
        print("Avg Train Loss: ", np.mean(train_loss))
        print("Avg Test Loss: ", np.mean(test_loss))
        print("Epoch Completed")

    return epoch_train_loss, epoch_test_loss


def predict(model, init_text, word2idx, idx2word, next_words=100):
    model.eval()
    vocab = list(word2idx.keys())
    words = init_text.split(' ')

    state_h, state_c = model.init()

    for i in range(next_words):
        x = torch.tensor([[word2idx[w] if w in vocab else 'unk' for w in words[i:]]])
        y_pred, state_h, state_c = model(x, (state_h, state_c))
        logits = y_pred[0][-1]
        prob = F.softmax(logits, dim=0).detach().numpy()
        word_idx = np.random.choice(len(logits), p=prob)
        words.append(idx2word[word_idx])

    return words


if __name__ == "__main__":

    model = LSTMModel()
    train_dataset = FriendsDataset(config.train_file)
    test_dataset = FriendsDataset(config.test_file)

    # train the mdoel
    train_loss, test_loss = train(model, train_dataset, test_dataset)

    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.savefig('./data/lm_v1.png')

    # checkout predictions from model

    with open(config.word2idx, 'rb') as handle:
        word2idx = pickle.load(handle)

    with open(config.idx2word, 'rb') as handle:
        idx2word = pickle.load(handle)
    


    init_text = 'hi chandler you look funny today'
    produced_text = predict(model, init_text, word2idx, idx2word)
    print(produced_text)

