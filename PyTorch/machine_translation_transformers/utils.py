import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

train_iter = Multi30k(split = "train", language_pair=("en", "de"))

def yield_tokens(train_iter):
    for data_sample in train_iter:
        #print(data_sample[0])
        print(get_tokenizer("spacy", language="de_core_news_sm")(data_sample[0]))

vocab_transform = {}
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']
#vocab_transform["en"] = build_vocab_from_iterator(yield_tokens(train_iter),min_freq=100,specials=special_symbols,special_first=True)

yield_tokens(train_iter)
#print(vocab_transform)

