import pickle
import numpy as np
import contractions
from nltk.tokenize import WordPunctTokenizer
from collections import Counter


# important params
max_seq_len = 32
vocab_size = 20000

# read data
folder_path = '/Users/sachin/Data/LM/'
fname = folder_path+'friends_script_edit_v2.txt'
with open(fname, 'r') as f:
    text = f.read()

# fix contractions
fixed_text = contractions.fix(text)
fixed_text = fixed_text.lower()

# tokenize
tokenizer = WordPunctTokenizer()
tokens = tokenizer.tokenize(fixed_text)

# create dictionary
token_freq = Counter(tokens)
top_tokens = token_freq.most_common(vocab_size-1)
tok_list = [j[0] for j in top_tokens]
word2idx = {tok: idx+1 for idx,tok in enumerate(tok_list)}
word2idx['UNK'] = 0
idx2word = {idx: word for word,idx in word2idx.items()}

# serialise text document
int_tokens = [word2idx[tok] if tok in tok_list 
                else word2idx['UNK'] for tok in tokens]

# create batches of data

n_samples = len(int_tokens)//(max_seq_len+1)

data = np.array(int_tokens[:n_samples*max_seq_len]).reshape(
        n_samples, max_seq_len)

# data = data.astype(np.int32)

# reshuffle rows
np.random.shuffle(data)

N = data.shape[0]

train_size = 0.80
train_idx = int(N*train_size)

train_data = data[:train_idx]
test_data = data[train_idx:]

# save data to disk
train_out = './data/friends32_20k_train.npy'
np.save(train_out, train_data)

test_out = './data/friends32_20k_test.npy'
np.save(test_out, test_data)

# save dictionary
with open('./data/word2idx.pickle', 'wb') as handle:
    pickle.dump(word2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/idx2word.pickle', 'wb') as handle:
    pickle.dump(idx2word, handle, protocol=pickle.HIGHEST_PROTOCOL)



