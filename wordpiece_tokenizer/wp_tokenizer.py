import json
import os
import time
from collections import Counter
from pathlib import Path

WP_PREFIX = "##"


def init_vocab(text: str) -> dict[str, int]:
    """create char levle vocab for given corpus"""
    vocab = {}
    word_list = text.split(" ")
    for word in word_list:
        for idx, ch in enumerate(word):
            if idx != 0:
                ch = WP_PREFIX + ch

            if ch not in vocab:
                vocab[ch] = 1
            else:
                vocab[ch] += 1

    return vocab


def word_tokenize(word: str, vocab: dict[str, int]) -> list[str]:
    """tokenize the word with given vocabulary"""
    stop_id = 1
    word_token_list = []
    while True:
        pattern = word[:stop_id]
        # print(pattern)
        if pattern in vocab:
            stop_id += 1
            if stop_id > len(word):
                word_token_list.append(pattern)
                break
        else:
            found_match = word[: stop_id - 1]
            word_token_list.append(found_match)
            # print(word_token_list)
            # time.sleep(1)
            word = word[stop_id - 1 :]
            if len(word) > 0:
                word = WP_PREFIX + word
                stop_id = 3
            else:
                break

    return word_token_list


def tokenize(text: str, vocab: dict[str, int]) -> list[str]:
    """tokenize text with given vocabulary"""
    tokenized_list = []
    word_list = text.split(" ")
    for word in word_list:
        tokenized_list += word_tokenize(word, vocab)

    return tokenized_list


def get_vocab_stats(tokenized_corpus: list[str]) -> str:
    """returns joint form of token pair with maximum score"""
    # token counter
    token_counter = Counter()
    token_counter.update(tokenized_corpus)

    token_pairs = []
    for i in range(len(tokenized_corpus) - 1):
        curr_word = tokenized_corpus[i]
        next_word = tokenized_corpus[i + 1]
        if not next_word.startswith("##"):
            pass
        else:
            token_pairs.append((curr_word, next_word))

    token_pair_counter = Counter()
    token_pair_counter.update(token_pairs)
    # print(token_pair_counter.most_common(100))

    max_token_pair = None
    max_token_score = -1
    for pair in token_pairs:
        freq_tok1 = token_counter.get(pair[0])
        freq_tok2 = token_counter.get(pair[1])
        freq_pair = token_pair_counter.get(pair)
        score = freq_pair / (freq_tok1 * freq_tok2)
        if score > max_token_score:
            max_token_score = score
            max_token_pair = pair

    # print(max_token_pair)

    joint_token = max_token_pair[0] + max_token_pair[1][2:]

    # [TODO] fix joint token logic

    return joint_token


def fit_tokenizer(text: str, max_vocab_size: int = 1000) -> dict[str, int]:
    """fit tokenizer on a text corpus"""
    text = text.lower()
    vocab = {}
    vocab = init_vocab(text)

    while len(vocab) < max_vocab_size:
        # tokenize with current vocab
        token_list = tokenize(text, vocab)
        joint_token = get_vocab_stats(token_list)
        print(joint_token)
        vocab[joint_token] = 1
        print(len(vocab))

    vocab = list(vocab.keys())
    vocab = {k: v for k, v in enumerate(vocab)}
    return vocab


if __name__ == "__main__":
    filename = "Downloads/Data/Lord_of_the_Rings.txt"
    file_path = os.path.join(Path.home(), filename)

    with open(file_path, "r") as f:
        text = f.read()[:100_000]

    # print(text[:100])

    vocab = fit_tokenizer(text)
    print(vocab)
    # print(len(vocab))

    # test word tokenize function
    # dummy_word = "cannibalization"
    # dummy_vocab = init_vocab(dummy_word)
    # dummy_vocab["##nn"] = 1
    # word_token_list = word_tokenize(dummy_word, dummy_vocab)
    # print("Word Tokenize Func Test: ")
    # print("Vocab: ", dummy_vocab)
    # print("Final Tokenization: ", word_token_list)
