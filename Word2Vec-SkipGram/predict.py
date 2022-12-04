# encoding=utf8
import io
import os
import pickle
import sys
import requests
from collections import OrderedDict
import math
import random
import numpy as np
import paddle
from paddle.nn import Embedding
import paddle.nn.functional as F
import paddle.nn as nn
from SkipGram import SkipGram


# # 读取
f_read = open('./my_model/word2id_freq.pkl', 'rb')
word2id_freq = pickle.load(f_read)
# print(word2id_freq)
vocab_size = len(word2id_freq)
f_read.close()

f_read = open('./my_model/word2id_dict.pkl', 'rb')
word2id_dict = pickle.load(f_read)
f_read.close()

f_read = open('./my_model/id2word_dict.pkl', 'rb')
id2word_dict = pickle.load(f_read)
f_read.close()


def get_similar_tokens(query_token, k, embed):
    W = embed.numpy()
    x = W[word2id_dict[query_token]]
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    flat = cos.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    for i in indices:
        print('for word %s, the similar word is %s' % (query_token, str(id2word_dict[i])))


# 定义一些训练过程中需要使用的超参数
embedding_size = 200
learning_rate = 0.001

# 通过我们定义的SkipGram类，来构造一个Skip-gram模型网络

skip_gram_model = SkipGram(vocab_size, embedding_size)

# 构造训练这个网络的优化器
adam = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=skip_gram_model.parameters())

# 加载网络模型和优化器模型
layer_state_dict = paddle.load("./my_model/text8.pdparams")
opt_state_dict = paddle.load("./my_model/adam.pdopt")

skip_gram_model.set_state_dict(layer_state_dict)
adam.set_state_dict(opt_state_dict)

# get_similar_tokens('movie', 5, skip_gram_model.embedding.weight)
# get_similar_tokens('one', 5, skip_gram_model.embedding.weight)
# get_similar_tokens('chip', 5, skip_gram_model.embedding.weight)
get_similar_tokens('she', 5, skip_gram_model.embedding.weight)
get_similar_tokens('dog', 5, skip_gram_model.embedding.weight)
get_similar_tokens('apple', 5, skip_gram_model.embedding.weight)
get_similar_tokens('beijing', 5, skip_gram_model.embedding.weight)