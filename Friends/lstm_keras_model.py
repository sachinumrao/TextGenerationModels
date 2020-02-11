# Import dependencies
import io
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Nadam


# Text data file
fname = "friends_script_editted.txt"

# Load data
with io.open(fname, encoding='utf-8') as f:
    raw_text = f.read().lower()

print("Raw Corpus Length: ", len(raw_text))

# Convert data to indices
print("Raw Characters")
raw_chars = sorted(list(set(raw_text)))
print("Total Characters: ", len(raw_chars))
print("Characters: ")
print(raw_chars)

# Remove unwanted characters
print("\nRemoving Unwanted Characters from Text Data...")
allowed_chars = ["\n", " ", ",", ".", "-", ":", 
                    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
                    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                    "u", "v", "w", "x", "y", "z"]

for ch in raw_chars:
    if ch not in allowed_chars:
        raw_text = raw_text.replace(ch, "")

text = raw_text

print(text[:32])
print("Processed Characters")
chars = sorted(list(set(text)))
print("Total Characters: ", len(chars))
print()
print("Characters: ")
print(chars)
print()

char2id = dict((c,i) for i,c in enumerate(chars))
id2char = dict((i,c) for i,c in enumerate(chars))

# Cut text into semi-redundant sequenes of certain max-length
print("Sequence Creation...")
maxlen = 64
step = 4
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i+maxlen])
    next_chars.append(text[i+maxlen])
print("Number of Sequences: ", len(sentences))


# Build models
model = Sequential()
model.add(Bidirectional(GRU(128, input_shape=(maxlen, len(chars)),
                        return_sequences=True)))

model.add(Bidirectional(GRU(512, return_sequences=True)))

model.add(Bidirectional(GRU(1024, return_sequences=True)))

model.add(GlobalAveragePooling1D())

model.add(Dense(1024, activation="relu"))

model.add(Dense(512, activation="relu"))

model.add(Dense(256, activation="relu"))

model.add(Dropout(0.05))

model.add(Dense(128, activation="relu"))

model.add(Dense(len(chars), activation="softmax"))


# Create optimizer
optimizer = Nadam()
model.compile(loss="categorical_crossentropy", optimizer=optimizer)

# Sample function
def sample(preds, temprature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temprature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Keras model callback
def on_epoch_end(epoch, _):
    if epoch%20 == 0:
        print()
        print("---------------------------- Generating Text After Epoch: %d" % epoch)

        start_index = random.randint(0, len(text) - maxlen -1)
        for diversity in [0.25, 0.5, 1.0, 1.25]:
            print("\n------------------------ Diversity: ", diversity)

            generated = ''
            sents = text[start_index: start_index+maxlen]
            generated += sents
            print("------------------------ Generating with seed: " + sents)
            sys.stdout.write(generated)

            for i in range(500):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t,char in enumerate(sents):
                    x_pred[0, t, char2id[char]] = 1.0

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = id2char[next_index]

                sents = sents[1:] + next_char
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

print_calback = LambdaCallback(on_epoch_end=on_epoch_end)

# Train the model


# Vectorization
data_batch = 50000
num_data_batch = len(sentences) // data_batch
print("Data Batches: ",num_data_batch)

num_epochs=2
print_model = True
for epoch in range(num_epochs):
    for j in range(num_data_batch):

        batch_sents = sentences[j*data_batch : (j+1)*data_batch]
        
        print("Vectorization...")
        x = np.zeros((len(batch_sents), maxlen, len(chars)))
        y = np.zeros((len(batch_sents), len(chars)))
        for i, sentence in enumerate(batch_sents):
            for t,char in enumerate(sentence):
                x[i, t, char2id[char]] = 1
            y[i, char2id[next_chars[i]]] = 1

        # Fit model
        model.fit(x,
                    y,
                    batch_size=128,
                    epochs=epoch+1,
                    initial_epoch=epoch,
                    shuffle=True,
                    callbacks=[print_calback])
        
        if print_model:
            print("\nModel Architecture: ")
            print(model.summary())
            print_model = False

    

