import os
from pathlib import Path

import pandas as pd
import spacy
from tqdm.auto import tqdm

# define sentence lenghts
MAX_CONTEXT_LENGTH = 512
WORD2TOKEN_FACTOR = 1.3
MAX_WORD_LENGTH = int(MAX_CONTEXT_LENGTH / WORD2TOKEN_FACTOR)

# data path
DATA_DIR = Path.home() / "Data"
INPUT_FILENAMES = ["lotr_part1.txt", "lotr_part2.txt", "lotr_part3.txt"]
OUTPUT_FILE_NAME = f"lotr_grouped_{MAX_CONTEXT_LENGTH}.csv"

# load spacy model
NLP = spacy.load("en_core_web_sm")


def groupby_text():
    # read data chunks from different txt files
    print("Reading files...")
    data_chunks = []
    for data_file in INPUT_FILENAMES:
        data_file_path = os.path.join(DATA_DIR, data_file)
        with open(data_file_path, "r") as f:
            data = f.read()

        data = data.split("\n")
        data_chunks += data

    # convert chunks to list of sentences
    print("Running sentence tokenizer...")
    sentence_list = []
    for chunk in tqdm(data_chunks, total=len(data_chunks)):
        chunk_doc = NLP(chunk)
        for item in chunk_doc.sents:
            sentence_list.append(str(item).strip())

    # group sentences while keeping word length below MAX_WORD_LENGTH
    print("Grouping sentences...")
    grouped_sentences_list = []
    curr_len = 0
    curr_group = []
    for sent in sentence_list:
        n_words = len(sent.split())
        if curr_len + n_words < MAX_WORD_LENGTH:
            curr_group.append(sent)
            curr_len += n_words
        else:
            grouped_sent = " ".join(curr_group)
            grouped_sentences_list.append(grouped_sent)
            curr_group = [sent]
            curr_len = n_words

    # write grouped text to output file
    print(f"Count of grouped sentences: {len(grouped_sentences_list)}")
    for i in range(5):
        print(grouped_sentences_list[i])
        print("-" * 120)

    print("Saving grouped sentences to disk...")
    df = pd.DataFrame({"Text_Chunk_LOTR": grouped_sentences_list})
    df.to_csv(OUTPUT_FILE_NAME, index=False)


if __name__ == "__main__":
    groupby_text()
