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

data = data.astype(np.int32)

# save data to disk
outfile = folder_path+'friends_tokenized_data.npy'
np.save(outfile, data)




