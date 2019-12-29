import csv
import numpy as np
import pandas as pd
from utils import *


def init_word_embedding(vocab, w2v_file, embed_size):
    word_embeddings = np.random.uniform(-1.0, 1.0, (len(vocab), embed_size)).astype(np.float32)
    if len(w2v_file) > 0:
        glove_embeddings = pd.read_table(w2v_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        for i, word in enumerate(vocab):
            if i == 0:
                word_embeddings[i] = np.zeros((embed_size,)).astype(np.float32)     # for <PAD> token
            else:
                try:
                    word_embeddings[i] = glove_embeddings.loc[word].values
                except KeyError:
                    word_embeddings[i] = np.random.uniform(-1.0, 1.0, size=(embed_size,)).astype(np.float32)
    return word_embeddings


def init_user_expertise(users, user_expertise_file, embed_size):
    user_expertise = np.random.uniform(-1.0, 1.0, (len(users), embed_size)).astype(np.float32)
    if len(user_expertise_file) > 0:
        user_expertise = load_from_pickle(user_expertise_file)
    return user_expertise


def init_user_styles(users, user_style_file, embed_size):
    user_styles = np.random.uniform(-1.0, 1.0, (len(users), embed_size)).astype(np.float32)
    if len(user_style_file) > 0:
        user_styles = load_from_pickle(user_style_file)
    return user_styles
