lstm_size = 32
embed_dim = 32 
n_layers = 1 
n_vocab = 20000
max_len = 32
lstm_dropout = 0.1 
fc_dropout = 0.1

batch_size = 32
learning_rate = 0.01
epochs = 5

train_file = './data/friends32_20k_train.npy'
test_file = './data/friends32_20k_test.npy'
word2idx_file = './data/word2idx.pickle'
idx2words_file = './data/idx2word.pickle'
