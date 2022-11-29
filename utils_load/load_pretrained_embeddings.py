import io
import numpy as np
import string


TEXT_COL, LABEL_COL = 'text', 'sentiment'


def load_vectors(location, datasets, tokenizer):
    fin = io.open(location, 'r', encoding='utf-8', newline='\n', errors='ignore')
    vocab = create_vocab(datasets, tokenizer)
    embedding = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] in vocab:
            embedding[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
    return embedding, vocab


def create_vocab(datasets, tokenizer):
    train_dataset = datasets["train"]
    dev_dataset = datasets["dev"]
    vocab = create_sub_vocab(train_dataset, tokenizer)
    vocab_sub = create_sub_vocab(dev_dataset, tokenizer)
    vocab = vocab + vocab_sub
    vocab = list(set(vocab))
    return vocab


def create_sub_vocab(dataset, tokenizer):
    vocab_sub = []
    for row in dataset.iterrows():
        text = row[1][TEXT_COL]
        for token in tokenizer(text.lower().translate(str.maketrans('', '', string.punctuation)).strip()):
            vocab_sub.append(token)
    return vocab_sub


def load_word_idx_maps(vocab):
    word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
    idx2word = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
    index = 4
    for word in vocab:
        word2idx[word] = index
        idx2word[index] = word
        index = index + 1
    return word2idx, idx2word


def load_lookup_table(embedding, word2idx):
    lookup_table = {}
    rows = cols = 0
    pretrained_vectors = []
    indexes = []
    for (word, vector) in embedding.items():
        index = word2idx[word]
        lookup_table[index] = vector
        indexes.append(index)
        pretrained_vectors.append(vector)
        (cols,) = vector.shape

    indexes, pretrained_vectors = zip(*sorted(zip(indexes, pretrained_vectors)))
    initial_vectors = np.random.randn(4, cols)
    pretrained_vectors = np.vstack((initial_vectors, pretrained_vectors))
    return lookup_table, pretrained_vectors
